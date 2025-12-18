"""
Query the RAG indices and return top-k most similar *literature* images.

This script expects the file names produced by `rag_build_index.py` and `rag_embed_text.py`:
  rag_index/
    literature_image_embeddings.npy
    literature_meta.jsonl
    literature_text_embeddings.npy   (optional)
    input_image_embeddings.npy
    input_meta.jsonl
    input_text_embeddings.npy        (optional)

Two modes:
  - image-only: alpha=1.0
  - hybrid:     alpha in (0,1) combines image cosine + caption cosine
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


def _load_meta(meta_path: Path) -> List[dict]:
    meta: List[dict] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


def _find_row_by_filename(meta: List[dict], filename: str) -> int:
    for i, m in enumerate(meta):
        if str(m.get("filename", "")) == filename:
            return i
    raise KeyError(f"filename {filename!r} not found in input_meta.jsonl")


def _imgcat_cmd() -> Optional[List[str]]:
    """
    Best-effort detection of a terminal image renderer.
    Returns argv for a command that accepts a file path as the last argument.
    """
    if shutil.which("imgcat"):
        return ["imgcat"]
    if shutil.which("kitty"):
        return ["kitty", "+kitten", "icat"]
    if shutil.which("wezterm"):
        return ["wezterm", "imgcat"]
    return None


def _try_show_image(path: Path) -> bool:
    cmd = _imgcat_cmd()
    if not cmd:
        return False
    try:
        subprocess.run([*cmd, str(path)], check=False)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", type=Path, default=Path("rag_index"))
    ap.add_argument(
        "--input-filename",
        type=str,
        required=True,
        help="Which input image to use as the query (e.g., input3.png). Must exist in rag_index/input_meta.jsonl",
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1.0, help="Weight on image similarity (1.0 = image only).")
    ap.add_argument("--only-micrographs", action="store_true", default=True)
    ap.add_argument("--include-non-micrographs", dest="only_micrographs", action="store_false")
    ap.add_argument(
        "--show-images",
        action="store_true",
        help="Try to display the query + retrieved images directly in the terminal (requires imgcat/kitty/wezterm).",
    )
    ap.add_argument(
        "--open-images",
        action="store_true",
        help="Open the query + retrieved images in the default image viewer (macOS: Preview).",
    )
    args = ap.parse_args()

    lit_img_embs = np.load(args.index_dir / "literature_image_embeddings.npy")
    lit_meta = _load_meta(args.index_dir / "literature_meta.jsonl")
    if lit_img_embs.shape[0] != len(lit_meta):
        raise ValueError("literature_image_embeddings.npy rows do not match literature_meta.jsonl lines")

    inp_img_embs = np.load(args.index_dir / "input_image_embeddings.npy")
    inp_meta = _load_meta(args.index_dir / "input_meta.jsonl")
    if inp_img_embs.shape[0] != len(inp_meta):
        raise ValueError("input_image_embeddings.npy rows do not match input_meta.jsonl lines")

    # Ensure normalized for cosine
    lit_img_embs = _normalize_rows(lit_img_embs.astype(np.float32, copy=False))
    inp_img_embs = _normalize_rows(inp_img_embs.astype(np.float32, copy=False))

    q_row = _find_row_by_filename(inp_meta, args.input_filename)
    q_img = _normalize_vec(inp_img_embs[q_row].astype(np.float32, copy=False))

    sim_img = lit_img_embs @ q_img

    sim_txt = None
    lit_txt_path = args.index_dir / "literature_text_embeddings.npy"
    inp_txt_path = args.index_dir / "input_text_embeddings.npy"
    if args.alpha < 1.0 and lit_txt_path.exists() and inp_txt_path.exists():
        lit_txt = _normalize_rows(np.load(lit_txt_path).astype(np.float32, copy=False))
        inp_txt = _normalize_rows(np.load(inp_txt_path).astype(np.float32, copy=False))
        q_txt = _normalize_vec(inp_txt[q_row].astype(np.float32, copy=False))
        sim_txt = lit_txt @ q_txt

    score = sim_img if sim_txt is None else (args.alpha * sim_img + (1.0 - args.alpha) * sim_txt)

    # Apply micrograph-only filtering if desired
    if args.only_micrographs:
        mask = np.array([bool(m.get("is_micrograph", True)) for m in lit_meta], dtype=bool)
        score = np.where(mask, score, -1e9)

    topk_idx = np.argsort(-score)[: args.top_k]

    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(topk_idx, start=1):
        m = lit_meta[int(idx)]
        results.append(
            {
                "rank": rank,
                "score": float(score[int(idx)]),
                "relpath": m.get("relpath"),
                "corpus_relpath": m.get("corpus_relpath"),
                "corpus_abspath": m.get("corpus_abspath"),
                "paper": m.get("paper"),
                "caption": m.get("caption", ""),
                "archived_to": m.get("archived_to"),
            }
        )

    # Optional: show images in terminal and/or open them in the system viewer.
    if args.show_images or args.open_images:
        q_img_path = Path(inp_meta[q_row].get("corpus_abspath") or "")
        if q_img_path.exists():
            print("\n=== QUERY IMAGE ===", file=sys.stderr)
            print(str(q_img_path), file=sys.stderr)
            if args.show_images:
                _try_show_image(q_img_path)
            if args.open_images:
                try:
                    subprocess.run(["open", str(q_img_path)], check=False)
                except Exception:
                    pass

        if results:
            print("\n=== TOP K RESULTS ===", file=sys.stderr)
        for r in results:
            p = Path(r.get("corpus_abspath") or "")
            if not p.exists():
                continue
            print(f"\n# rank={r.get('rank')} score={r.get('score')}", file=sys.stderr)
            print(str(p), file=sys.stderr)
            if args.show_images:
                _try_show_image(p)
            if args.open_images:
                try:
                    subprocess.run(["open", str(p)], check=False)
                except Exception:
                    pass

    print(
        json.dumps(
            {"input_filename": args.input_filename, "top_k": args.top_k, "alpha": args.alpha, "results": results},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


