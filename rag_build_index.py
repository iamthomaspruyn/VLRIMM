"""
Build unified RAG indices (NO API calls).

Inputs:
  - dinov3_nested_dict.pkl         (paper -> image_key -> image_embedding)
  - vlm_results.jsonl              (paper images captions/metadata)
  - input_image_embeddings.pkl     (input image embeddings; supports dict or nested dict)
  - input_vlm_results.jsonl        (input images captions/metadata)

Assumes you created a unified corpus folder:
  RAG_Corpus/
    Inputs/<file>
    Literature/<paper>/images_processed/<file>

Outputs (default: ./rag_index/):
  - literature_meta.jsonl
  - literature_image_embeddings.npy
  - input_meta.jsonl
  - input_image_embeddings.npy
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _norm_slashes(s: str) -> str:
    return s.replace("\\", "/")


def _as_float32_row(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32, copy=False)


def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _guess_lit_relpath(paper: str, image_key: str) -> str:
    """
    Normalize keys from DINO embedding dict to match vlm_results.jsonl relpaths:
      <paper>/images_processed/<filename>
    """
    k = _norm_slashes(str(image_key).strip().lstrip("/"))
    if "images_processed/" in k:
        if k.startswith(f"{paper}/"):
            return k
        if k.startswith("images_processed/"):
            return f"{paper}/{k}"
        return f"{paper}/{k}"
    if "/" in k:
        fname = k.split("/")[-1]
        return f"{paper}/images_processed/{fname}"
    return f"{paper}/images_processed/{k}"


def _guess_input_relpath(key: str) -> str:
    # want just filename to match input_vlm_results.jsonl (image_relpath is "input1.png")
    k = _norm_slashes(str(key).strip().lstrip("/"))
    return k.split("/")[-1]


def _iter_flat_input_embeddings(obj: Any) -> Iterable[Tuple[str, Any]]:
    """
    Support:
      - dict[key] = embedding
      - dict[group][key] = embedding  => yields "group::key"
    """
    if not isinstance(obj, dict):
        return
    for k, v in obj.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                yield f"{k}::{kk}", vv
        else:
            yield str(k), v


@dataclass
class LitItem:
    relpath: str
    paper: str
    filename: str
    image_embedding: np.ndarray
    caption: str
    is_micrograph: Optional[bool]
    archived_to: Optional[str]
    corpus_relpath: str
    corpus_abspath: str


@dataclass
class InputItem:
    relpath: str  # filename
    filename: str
    source_key: str
    image_embedding: np.ndarray
    caption: str
    is_micrograph: Optional[bool]
    corpus_relpath: str
    corpus_abspath: str


def build_literature_items(
    *,
    dinov3_pkl: Path,
    vlm_results_jsonl: Path,
    corpus_root: Path,
    normalize: bool,
) -> List[LitItem]:
    din = _load_pickle(dinov3_pkl)
    if not isinstance(din, dict):
        raise TypeError("dinov3_nested_dict.pkl must be dict(paper -> dict(image -> embedding))")

    vlm_by_rel: Dict[str, dict] = {}
    for rec in _iter_jsonl(vlm_results_jsonl):
        rel = rec.get("image_relpath")
        if isinstance(rel, str):
            vlm_by_rel[_norm_slashes(rel)] = rec

    out: List[LitItem] = []
    for paper, inner in din.items():
        if not isinstance(inner, dict):
            continue
        paper_s = str(paper)
        for image_key, emb in inner.items():
            rel = _guess_lit_relpath(paper_s, str(image_key))
            vrec = vlm_by_rel.get(rel)
            caption = (vrec.get("description") if isinstance(vrec, dict) else "") or ""
            is_micro = (vrec.get("is_micrograph") if isinstance(vrec, dict) else None)
            archived_to = (vrec.get("archived_to") if isinstance(vrec, dict) else None)

            row = _as_float32_row(emb)
            if normalize:
                row = _l2_norm(row)

            filename = Path(rel).name
            corpus_rel = f"Literature/{rel}"
            corpus_abs = str((corpus_root / corpus_rel).resolve())

            out.append(
                LitItem(
                    relpath=rel,
                    paper=paper_s,
                    filename=filename,
                    image_embedding=row,
                    caption=str(caption),
                    is_micrograph=is_micro if isinstance(is_micro, bool) else None,
                    archived_to=str(archived_to) if isinstance(archived_to, str) else None,
                    corpus_relpath=corpus_rel,
                    corpus_abspath=corpus_abs,
                )
            )
    return out


def build_input_items(
    *,
    input_embeddings_pkl: Path,
    input_vlm_jsonl: Path,
    corpus_root: Path,
    normalize: bool,
) -> List[InputItem]:
    obj = _load_pickle(input_embeddings_pkl)

    vlm_by_rel: Dict[str, dict] = {}
    for rec in _iter_jsonl(input_vlm_jsonl):
        rel = rec.get("image_relpath")
        if isinstance(rel, str):
            vlm_by_rel[_norm_slashes(rel)] = rec

    out: List[InputItem] = []
    for source_key, emb in _iter_flat_input_embeddings(obj):
        rel = _guess_input_relpath(source_key)
        vrec = vlm_by_rel.get(rel)
        caption = (vrec.get("description") if isinstance(vrec, dict) else "") or ""
        is_micro = (vrec.get("is_micrograph") if isinstance(vrec, dict) else None)

        row = _as_float32_row(emb)
        if normalize:
            row = _l2_norm(row)

        filename = Path(rel).name
        corpus_rel = f"Inputs/{filename}"
        corpus_abs = str((corpus_root / corpus_rel).resolve())

        out.append(
            InputItem(
                relpath=rel,
                filename=filename,
                source_key=str(source_key),
                image_embedding=row,
                caption=str(caption),
                is_micrograph=is_micro if isinstance(is_micro, bool) else None,
                corpus_relpath=corpus_rel,
                corpus_abspath=corpus_abs,
            )
        )
    return out


def _write_split(
    *,
    out_dir: Path,
    split: str,
    image_embeddings: np.ndarray,
    meta_rows: List[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{split}_image_embeddings.npy", image_embeddings)
    with (out_dir / f"{split}_meta.jsonl").open("w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("rag_index"))
    ap.add_argument("--corpus-root", type=Path, default=Path("RAG_Corpus"))

    ap.add_argument("--dinov3-pkl", type=Path, default=Path("dinov3_nested_dict.pkl"))
    ap.add_argument("--vlm-results", type=Path, default=Path("vlm_results.jsonl"))

    ap.add_argument("--input-embeddings-pkl", type=Path, default=Path("input_image_embeddings.pkl"))
    ap.add_argument("--input-vlm-results", type=Path, default=Path("input_vlm_results.jsonl"))

    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    args = ap.parse_args()

    lit_items = build_literature_items(
        dinov3_pkl=args.dinov3_pkl,
        vlm_results_jsonl=args.vlm_results,
        corpus_root=args.corpus_root,
        normalize=args.normalize,
    )
    if lit_items:
        D = lit_items[0].image_embedding.shape[0]
        lit_mat = np.zeros((len(lit_items), D), dtype=np.float32)
        for i, it in enumerate(lit_items):
            lit_mat[i, :] = it.image_embedding
    else:
        lit_mat = np.zeros((0, 0), dtype=np.float32)
    lit_meta = [
        {
            "relpath": it.relpath,
            "paper": it.paper,
            "filename": it.filename,
            "caption": it.caption,
            "is_micrograph": it.is_micrograph,
            "archived_to": it.archived_to,
            "corpus_relpath": it.corpus_relpath,
            "corpus_abspath": it.corpus_abspath,
        }
        for it in lit_items
    ]
    _write_split(out_dir=args.out_dir, split="literature", image_embeddings=lit_mat, meta_rows=lit_meta)

    inp_items = build_input_items(
        input_embeddings_pkl=args.input_embeddings_pkl,
        input_vlm_jsonl=args.input_vlm_results,
        corpus_root=args.corpus_root,
        normalize=args.normalize,
    )
    if inp_items:
        D2 = inp_items[0].image_embedding.shape[0]
        inp_mat = np.zeros((len(inp_items), D2), dtype=np.float32)
        for i, it in enumerate(inp_items):
            inp_mat[i, :] = it.image_embedding
    else:
        inp_mat = np.zeros((0, 0), dtype=np.float32)
    inp_meta = [
        {
            "relpath": it.relpath,
            "filename": it.filename,
            "source_key": it.source_key,
            "caption": it.caption,
            "is_micrograph": it.is_micrograph,
            "corpus_relpath": it.corpus_relpath,
            "corpus_abspath": it.corpus_abspath,
        }
        for it in inp_items
    ]
    _write_split(out_dir=args.out_dir, split="input", image_embeddings=inp_mat, meta_rows=inp_meta)

    print(f"Wrote indices to {args.out_dir}")
    print(f"  literature rows: {len(lit_items)}")
    print(f"  input rows:      {len(inp_items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


