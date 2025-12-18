"""
Two-stage VLM pipeline for figure images (micrograph filter + captioning).

Pipeline:
1) Classify each image as a materials-science micrograph vs not (cheap model).
2) If micrograph, generate a brief (3–4 sentence) materials-science style description (stronger model).
3) Optionally, archive non-micrographs into a separate folder while preserving relative structure.

This script is resumable:
- It appends one JSON record per image to --results-file
- On re-runs, it skips images already present in that JSONL (by `image_relpath`)

Supported input layouts:
- Docling layout (auto-detected):
    docling_output/<paper_name>/images_processed/*.(png|jpg|jpeg|webp)
  In this mode, only images under `images_processed/` folders are considered.

- Flat folder layout:
    input_images/*.(png|jpg|jpeg|webp)
  In this mode, all images under the root are considered.

Examples
--------
Docling images:
  export OPENAI_API_KEY=...
  python vlm_caption.py --images-root docling_output --results-file vlm_results.jsonl --archive-root archive_non_micrographs

Input images:
  export OPENAI_API_KEY=...
  python vlm_caption.py --images-root input_images --results-file input_vlm_results.jsonl --no-require-images-processed
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from openai import OpenAI


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _data_url_for_image(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _load_processed(results_file: Path) -> set[str]:
    processed: set[str] = set()
    if not results_file.exists():
        return processed
    with results_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel = rec.get("image_relpath")
            if isinstance(rel, str):
                processed.add(rel)
    return processed


def _iter_images(images_root: Path, *, require_images_processed: bool) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for p in images_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if require_images_processed and p.parent.name != "images_processed":
            continue
        yield p


def _auto_require_images_processed(images_root: Path) -> bool:
    """
    Heuristic:
    - If there exists at least one image under a directory named `images_processed`, assume Docling layout.
    - Otherwise treat as a generic image folder.
    """
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and p.parent.name == "images_processed":
            return True
    return False


def _paper_from_relpath(rel: str) -> Optional[str]:
    # expected for docling: <paper>/images_processed/<file>
    parts = Path(rel).parts
    return parts[0] if len(parts) >= 3 else None


CLASSIFY_SYSTEM = (
    "You are a careful assistant for materials-science literature mining. "
    "Your job is ONLY to classify whether an image is a micrograph."
)

CLASSIFY_USER = """Decide if the image is a materials-science micrograph.

Micrographs include: SEM, TEM, STEM, optical microscopy of microstructure,
EBSD/IPF maps, etched microstructures, fracture surface micrographs.

Not micrographs include: plots/graphs, schematic diagrams, flowcharts, tables, photos of equipment,
screenshots, molecule drawings, crystal structure cartoons, maps unrelated to microscopy, and text-only figures.

Return ONLY valid JSON with keys:
- is_micrograph: boolean
- confidence: number between 0 and 1
- rationale: one short sentence
"""

CAPTION_SYSTEM = (
    "You are a meticulous materials scientist writing brief descriptions of microscopy images. "
    "Do not guess magnification, composition, phase names, or processing conditions unless clearly visible."
)

CAPTION_USER = """Write a concise 3–4 sentence description as a materials scientist.

- Describe the apparent microstructure: grains, phases/contrast regions, porosity, cracks, precipitates, lamellae, etc.
- Mention morphology and spatial distribution (e.g., equiaxed/columnar, dendritic, network-like, layered).
- If a scale bar or units (µm/nm) are visible, mention that a scale bar is present, but do NOT infer magnification.
- Avoid speculation: if details are unclear, say so plainly.

Return ONLY valid JSON with key:
- description: string (3–4 sentences)
"""


def classify_micrograph(
    client: OpenAI,
    *,
    model: str,
    image_path: Path,
) -> Tuple[bool, float, str, Optional[str]]:
    data_url = _data_url_for_image(image_path)
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": CLASSIFY_USER},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        temperature=0,
    )
    raw = (resp.output_text or "").strip()
    obj = _extract_json(raw)
    if not obj:
        return False, 0.0, "Failed to parse JSON from model output.", raw
    is_micro = bool(obj.get("is_micrograph", False))
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    rationale = str(obj.get("rationale", "")).strip()
    return is_micro, max(0.0, min(conf, 1.0)), rationale, None


def caption_micrograph(
    client: OpenAI,
    *,
    model: str,
    image_path: Path,
) -> Tuple[str, Optional[str]]:
    data_url = _data_url_for_image(image_path)
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": CAPTION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": CAPTION_USER},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        temperature=0.2,
    )
    raw = (resp.output_text or "").strip()
    obj = _extract_json(raw)
    if not obj:
        return "", raw
    desc = str(obj.get("description", "")).strip()
    return desc, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", type=Path, required=True)
    ap.add_argument("--results-file", type=Path, required=True)
    ap.add_argument("--archive-root", type=Path, default=Path("archive_non_micrographs"))
    ap.add_argument("--classify-model", type=str, default="gpt-4o-mini")
    ap.add_argument("--caption-model", type=str, default="gpt-4o")

    ap.add_argument(
        "--require-images-processed",
        action="store_true",
        default=None,
        help="Only process images inside folders named 'images_processed'. Default: auto-detect.",
    )
    ap.add_argument(
        "--no-require-images-processed",
        dest="require_images_processed",
        action="store_false",
        help="Process all images under --images-root (recursively).",
    )

    ap.add_argument("--move-non-micrographs", action="store_true", default=True)
    ap.add_argument("--no-move-non-micrographs", dest="move_non_micrographs", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="Do not call the API and do not move files.")
    ap.add_argument("--max-images", type=int, default=0, help="If >0, limit number of images processed.")
    args = ap.parse_args()

    if not args.images_root.exists():
        print(f"images root not found: {args.images_root}", file=sys.stderr)
        return 2

    require_images_processed: bool
    if args.require_images_processed is None:
        require_images_processed = _auto_require_images_processed(args.images_root)
    else:
        require_images_processed = bool(args.require_images_processed)

    processed = _load_processed(args.results_file)
    images = sorted(_iter_images(args.images_root, require_images_processed=require_images_processed))
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    to_process = []
    for p in images:
        rel = str(p.relative_to(args.images_root))
        if rel not in processed:
            to_process.append((p, rel))

    print(
        f"Found {len(images)} images; already processed {len(processed)}; remaining {len(to_process)} "
        f"(require_images_processed={require_images_processed})"
    )

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    client = OpenAI() if not args.dry_run else None
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    if args.move_non_micrographs:
        args.archive_root.mkdir(parents=True, exist_ok=True)

    with args.results_file.open("a", encoding="utf-8") as out:
        for img_path, rel in to_process:
            paper = _paper_from_relpath(rel) if require_images_processed else None
            rec: Dict[str, Any] = {
                "ts_utc": _utc_now_iso(),
                "image_relpath": rel,
                "image_abspath": str(img_path.resolve()),
                "paper": paper,
                "classify_model": args.classify_model,
                "caption_model": args.caption_model,
            }

            if args.dry_run:
                rec.update(
                    {
                        "is_micrograph": False,
                        "confidence": 0.0,
                        "rationale": "dry-run",
                        "description": "",
                        "archived_to": None,
                    }
                )
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                continue

            assert client is not None
            try:
                is_micro, conf, rationale, classify_error = classify_micrograph(
                    client, model=args.classify_model, image_path=img_path
                )
                rec.update(
                    {
                        "is_micrograph": is_micro,
                        "confidence": conf,
                        "rationale": rationale,
                        "classify_error": classify_error,
                    }
                )

                description = ""
                caption_error = None
                if is_micro:
                    description, caption_error = caption_micrograph(
                        client, model=args.caption_model, image_path=img_path
                    )
                rec.update({"description": description, "caption_error": caption_error})

                archived_to = None
                if (not is_micro) and args.move_non_micrographs:
                    dest = args.archive_root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(img_path), str(dest))
                    archived_to = str(dest)
                rec["archived_to"] = archived_to

            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

    print(f"Wrote results to {args.results_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


