"""
Compute OpenAI text embeddings for captions in a meta.jsonl file (resumable).

Usage:
  export OPENAI_API_KEY=...
  python rag_embed_text.py --meta rag_index/literature_meta.jsonl --out rag_index/literature_text_embeddings.npy
  python rag_embed_text.py --meta rag_index/input_meta.jsonl      --out rag_index/input_text_embeddings.npy

Notes:
  - Resumable via a cache JSONL next to --out (or --cache-jsonl)
  - Embeddings are L2-normalized by default (cosine-ready)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from openai import OpenAI


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _load_cache(cache_jsonl: Path) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    if not cache_jsonl.exists():
        return out
    for rec in _iter_jsonl(cache_jsonl):
        try:
            idx = int(rec.get("row_index"))
        except Exception:
            continue
        out[idx] = rec
    return out


def _batched(seq: List[Tuple[int, str, str]], n: int) -> Iterable[List[Tuple[int, str, str]]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--text-field", type=str, default="caption")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-jsonl", type=Path, default=None)
    ap.add_argument("--model", type=str, default="text-embedding-3-large")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.add_argument("--sleep-on-rate-limit", type=float, default=2.0)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    cache_jsonl = args.cache_jsonl or args.out.with_suffix(".cache.jsonl")
    cache_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    meta_rows = list(_iter_jsonl(args.meta))
    cache = _load_cache(cache_jsonl)

    to_embed: List[Tuple[int, str, str]] = []
    for i, row in enumerate(meta_rows):
        text = str(row.get(args.text_field, "") or "").strip()
        th = _sha256(f"{args.model}\n{text}")
        crec = cache.get(i)
        if (
            crec
            and crec.get("model") == args.model
            and isinstance(crec.get("text_hash"), str)
            and crec.get("text_hash") == th
            and isinstance(crec.get("embedding"), list)
            and len(crec.get("embedding")) > 0
        ):
            continue
        to_embed.append((i, text, th))

    client = OpenAI()

    emb_dim: Optional[int] = None
    for rec in cache.values():
        emb = rec.get("embedding")
        if isinstance(emb, list) and len(emb) > 0:
            emb_dim = len(emb)
            break

    def _embed_texts(texts: List[str]) -> List[List[float]]:
        for attempt in range(8):
            try:
                resp = client.embeddings.create(model=args.model, input=texts)
                return [d.embedding for d in resp.data]
            except Exception:
                time.sleep(args.sleep_on_rate_limit * (attempt + 1))
        resp = client.embeddings.create(model=args.model, input=texts)
        return [d.embedding for d in resp.data]

    with cache_jsonl.open("a", encoding="utf-8") as out:
        for batch in _batched(to_embed, max(1, args.batch_size)):
            idxs = [i for (i, _t, _h) in batch]
            texts = [t for (_i, t, _h) in batch]
            hashes = [h for (_i, _t, h) in batch]

            need = [(i, t, h) for (i, t, h) in zip(idxs, texts, hashes) if t]
            if need:
                need_idxs = [i for (i, _t, _h) in need]
                need_texts = [t for (_i, t, _h) in need]
                need_hashes = [h for (_i, _t, h) in need]
                embs = _embed_texts(need_texts)
                if emb_dim is None and embs:
                    emb_dim = len(embs[0])
                for i, h, e in zip(need_idxs, need_hashes, embs):
                    rec = {"row_index": i, "model": args.model, "text_hash": h, "embedding": e}
                    cache[i] = rec
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()

            for i, t, h in batch:
                if t:
                    continue
                rec = {"row_index": i, "model": args.model, "text_hash": h, "embedding": []}
                cache[i] = rec
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

    if emb_dim is None:
        np.save(args.out, np.zeros((len(meta_rows), 0), dtype=np.float32))
        print(f"Wrote {args.out} shape={(len(meta_rows), 0)} (no non-empty captions)")
        return 0

    mat = np.zeros((len(meta_rows), emb_dim), dtype=np.float32)
    for i in range(len(meta_rows)):
        rec = cache.get(i)
        emb = rec.get("embedding") if isinstance(rec, dict) else None
        if isinstance(emb, list) and len(emb) == emb_dim:
            mat[i, :] = np.asarray(emb, dtype=np.float32)

    if args.normalize:
        mat = _normalize_rows(mat)

    np.save(args.out, mat)
    print(f"Wrote {args.out} shape={mat.shape} cache={cache_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


