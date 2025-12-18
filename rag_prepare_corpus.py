"""
Prepare a unified on-disk corpus folder for RAG.

Goal:
  RAG_Corpus/
    Inputs/                 (query/input images)
    Literature/<paper>/images_processed/<file>   (paper images)

By default this uses symlinks (fast, no duplication). You can also use --mode copy.

It also writes a manifest JSONL so each corpus file can be traced back to its original source.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_images(root: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _safe_rel_under(base: Path, path: Path) -> str:
    return str(path.relative_to(base)).replace("\\", "/")


def _ensure_link_or_copy(*, src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if mode == "symlink":
        # Use relative symlinks when possible to keep corpus movable.
        try:
            rel_src = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel_src)
        except Exception:
            dst.symlink_to(src)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docling-root", type=Path, default=Path("docling_output"))
    ap.add_argument("--input-images-dir", type=Path, default=Path("input_images"))
    ap.add_argument("--corpus-root", type=Path, default=Path("RAG_Corpus"))
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files in the corpus.")
    ap.add_argument("--manifest", type=Path, default=Path("RAG_Corpus/manifest.jsonl"))
    ap.add_argument(
        "--include-paper-text",
        action="store_true",
        default=True,
        help="Also add each paper's docling_output/<paper>/text.txt into the corpus.",
    )
    ap.add_argument("--no-include-paper-text", dest="include_paper_text", action="store_false")
    ap.add_argument(
        "--paper-text-name",
        type=str,
        default="{paper}.txt",
        help="Destination filename for each paper's text (format string; supports {paper}).",
    )
    args = ap.parse_args()

    literature_dst_root = args.corpus_root / "Literature"
    inputs_dst_root = args.corpus_root / "Inputs"
    args.corpus_root.mkdir(parents=True, exist_ok=True)
    literature_dst_root.mkdir(parents=True, exist_ok=True)
    inputs_dst_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_f = args.manifest.open("a", encoding="utf-8")

    # ---- Inputs ----
    for src in _iter_images(args.input_images_dir):
        rel = _safe_rel_under(args.input_images_dir, src)  # usually just filename
        dst = inputs_dst_root / rel
        _ensure_link_or_copy(src=src, dst=dst, mode=args.mode, overwrite=args.overwrite)
        manifest_f.write(
            json.dumps(
                {
                    "ts_utc": _utc_now_iso(),
                    "kind": "input",
                    "src": str(src.resolve()),
                    "dst": str(dst.resolve()),
                    "dst_rel": str(dst.relative_to(args.corpus_root)).replace("\\", "/"),
                    "mode": args.mode,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    # ---- Literature ----
    # Preserve your existing paper structure: <paper>/images_processed/<file>
    if args.docling_root.exists():
        seen_papers_with_text: set[str] = set()
        for src in _iter_images(args.docling_root):
            # only include images under a paper's images_processed folder
            if src.parent.name != "images_processed":
                continue
            try:
                rel = _safe_rel_under(args.docling_root, src)  # <paper>/images_processed/<file>
            except Exception:
                continue
            dst = literature_dst_root / rel  # Literature/<paper>/images_processed/<file>
            _ensure_link_or_copy(src=src, dst=dst, mode=args.mode, overwrite=args.overwrite)
            manifest_f.write(
                json.dumps(
                    {
                        "ts_utc": _utc_now_iso(),
                        "kind": "literature_image",
                        "src": str(src.resolve()),
                        "dst": str(dst.resolve()),
                        "dst_rel": str(dst.relative_to(args.corpus_root)).replace("\\", "/"),
                        "mode": args.mode,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            # Also add per-paper text.txt once per paper
            if args.include_paper_text:
                paper = Path(rel).parts[0] if len(Path(rel).parts) >= 3 else None
                if paper and paper not in seen_papers_with_text:
                    text_src = args.docling_root / paper / "text.txt"
                    if text_src.exists() and text_src.is_file():
                        text_name = args.paper_text_name.format(paper=paper)
                        text_dst = literature_dst_root / paper / text_name
                        _ensure_link_or_copy(
                            src=text_src, dst=text_dst, mode=args.mode, overwrite=args.overwrite
                        )
                        manifest_f.write(
                            json.dumps(
                                {
                                    "ts_utc": _utc_now_iso(),
                                    "kind": "literature_text",
                                    "src": str(text_src.resolve()),
                                    "dst": str(text_dst.resolve()),
                                    "dst_rel": str(text_dst.relative_to(args.corpus_root)).replace("\\", "/"),
                                    "mode": args.mode,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    seen_papers_with_text.add(paper)

    manifest_f.close()
    print(f"Corpus prepared at: {args.corpus_root} (mode={args.mode})")
    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


