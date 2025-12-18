"""
Bundle an index split (meta + image embeddings + text embeddings) into a single .pkl.

Example:
  python rag_pack_index.py --index-dir rag_index --split literature
  python rag_pack_index.py --index-dir rag_index --split input
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np


def _load_meta(meta_path: Path) -> List[dict]:
    meta: List[dict] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", type=Path, default=Path("rag_index"))
    ap.add_argument("--split", choices=["literature", "input"], required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    meta_path = args.index_dir / f"{args.split}_meta.jsonl"
    img_path = args.index_dir / f"{args.split}_image_embeddings.npy"
    txt_path = args.index_dir / f"{args.split}_text_embeddings.npy"

    meta = _load_meta(meta_path)
    img = np.load(img_path)
    txt: Optional[np.ndarray] = None
    if txt_path.exists():
        txt = np.load(txt_path)

    if img.shape[0] != len(meta):
        raise ValueError(f"{img_path.name} rows ({img.shape[0]}) != meta rows ({len(meta)})")
    if txt is not None and txt.shape[0] != len(meta):
        raise ValueError(f"{txt_path.name} rows ({txt.shape[0]}) != meta rows ({len(meta)})")

    out_path = args.out or (args.index_dir / f"{args.split}_index.pkl")
    payload = {"split": args.split, "meta": meta, "image_embeddings": img, "text_embeddings": txt}
    with out_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Wrote {out_path} rows={len(meta)} img={tuple(img.shape)} txt={(None if txt is None else tuple(txt.shape))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


