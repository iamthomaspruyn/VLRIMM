"""
Textual RAG over the papers corresponding to the top-K retrieved images.

Workflow
--------
1) Retrieve top-K literature images for a query input image (using `rag_index/` files).
2) Collect unique paper IDs from those retrieved images.
3) For those papers only:
   - load paper text from `RAG_Corpus/Literature/<paper>/<paper>.txt` (fallback to `text.txt`)
   - chunk the text
   - embed chunks with OpenAI embeddings (cached on disk)
4) Embed the user question and retrieve top-N chunks by cosine similarity.
5) Produce an answer grounded in the retrieved chunks (with paper citations).

This avoids embedding/chunking the entire corpus up front: it only processes the subset of papers
needed for the current query (fast + cheaper for hackathon scale).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from openai import OpenAI


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _find_row_by_filename(meta: List[dict], filename: str) -> int:
    for i, m in enumerate(meta):
        if str(m.get("filename", "")) == filename:
            return i
    raise KeyError(f"filename {filename!r} not found in input_meta.jsonl")


def _load_meta(meta_path: Path) -> List[dict]:
    return list(_iter_jsonl(meta_path))


def retrieve_top_images(
    *,
    index_dir: Path,
    input_filename: str,
    top_k: int,
    alpha: float,
    only_micrographs: bool,
) -> Tuple[dict, List[dict]]:
    """
    Returns: (query_meta_row, topk_results)
    """
    lit_img = np.load(index_dir / "literature_image_embeddings.npy").astype(np.float32, copy=False)
    lit_meta = _load_meta(index_dir / "literature_meta.jsonl")
    if lit_img.shape[0] != len(lit_meta):
        raise ValueError("literature embeddings rows do not match literature_meta.jsonl")

    inp_img = np.load(index_dir / "input_image_embeddings.npy").astype(np.float32, copy=False)
    inp_meta = _load_meta(index_dir / "input_meta.jsonl")
    if inp_img.shape[0] != len(inp_meta):
        raise ValueError("input embeddings rows do not match input_meta.jsonl")

    lit_img = _normalize_rows(lit_img)
    inp_img = _normalize_rows(inp_img)

    q_row = _find_row_by_filename(inp_meta, input_filename)
    q_img = _normalize_vec(inp_img[q_row])

    sim_img = lit_img @ q_img

    score = sim_img
    lit_txt_path = index_dir / "literature_text_embeddings.npy"
    inp_txt_path = index_dir / "input_text_embeddings.npy"
    if alpha < 1.0 and lit_txt_path.exists() and inp_txt_path.exists():
        lit_txt = _normalize_rows(np.load(lit_txt_path).astype(np.float32, copy=False))
        inp_txt = _normalize_rows(np.load(inp_txt_path).astype(np.float32, copy=False))
        if lit_txt.shape[0] != len(lit_meta) or inp_txt.shape[0] != len(inp_meta):
            raise ValueError("text embeddings rows do not match meta rows")
        q_txt = _normalize_vec(inp_txt[q_row])
        sim_txt = lit_txt @ q_txt
        score = alpha * sim_img + (1.0 - alpha) * sim_txt

    if only_micrographs:
        mask = np.array([bool(m.get("is_micrograph", True)) for m in lit_meta], dtype=bool)
        score = np.where(mask, score, -1e9)

    topk_idx = np.argsort(-score)[:top_k]
    results: List[dict] = []
    for rank, idx in enumerate(topk_idx, start=1):
        m = lit_meta[int(idx)]
        results.append(
            {
                "rank": rank,
                "score": float(score[int(idx)]),
                "paper": m.get("paper"),
                "corpus_abspath": m.get("corpus_abspath"),
                "corpus_relpath": m.get("corpus_relpath"),
                "caption": m.get("caption", ""),
            }
        )

    return inp_meta[q_row], results


def load_paper_text(corpus_root: Path, paper: str) -> str:
    paper_dir = corpus_root / "Literature" / paper
    p1 = paper_dir / f"{paper}.txt"
    p2 = paper_dir / "text.txt"
    if p1.exists():
        return p1.read_text(encoding="utf-8", errors="ignore")
    if p2.exists():
        return p2.read_text(encoding="utf-8", errors="ignore")
    raise FileNotFoundError(f"No text file found for paper {paper!r} under {paper_dir}")


def _strip_references_section(markdown: str) -> str:
    """
    Remove the trailing reference section if present.
    Common Docling markdown pattern: a top-level section '## References' near the end.
    """
    m = re.search(r"(?m)^\s*##\s+References\s*$", markdown)
    if not m:
        return markdown
    return markdown[: m.start()].rstrip()


def _strip_docling_noise(markdown: str) -> str:
    # Remove placeholders that do not help retrieval.
    markdown = re.sub(r"(?m)^\s*<!--\s*image\s*-->\s*$", "", markdown)
    return markdown


def split_markdown_h2_sections(markdown: str) -> List[Tuple[str, str]]:
    """
    Split markdown into (section_title, section_body) based on '## ' headings.
    The heading line itself is not included in the body; it is returned as the title.
    """
    lines = markdown.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    cur_title = "Preamble"
    cur_body: List[str] = []

    h2 = re.compile(r"^\s*##\s+(.*\S)\s*$")
    for ln in lines:
        m = h2.match(ln)
        if m:
            sections.append((cur_title, cur_body))
            cur_title = m.group(1).strip()
            cur_body = []
        else:
            cur_body.append(ln)
    sections.append((cur_title, cur_body))

    out: List[Tuple[str, str]] = []
    for title, body_lines in sections:
        out.append((title, "\n".join(body_lines).strip()))
    return out


def chunk_section_text(section_text: str, *, chunk_chars: int, overlap_chars: int) -> List[str]:
    """
    Chunk a single section into ~chunk_chars blocks, preserving paragraph boundaries.
    """
    norm = re.sub(r"[ \t]+", " ", section_text)
    paras = [p.strip() for p in re.split(r"\n\s*\n+", norm) if p.strip()]
    chunks: List[str] = []

    buf: List[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunks.append("\n\n".join(buf).strip())
        buf, buf_len = [], 0

    for p in paras:
        if len(p) > chunk_chars:
            flush()
            for i in range(0, len(p), max(1, chunk_chars - overlap_chars)):
                chunks.append(p[i : i + chunk_chars].strip())
            continue

        if buf_len + len(p) + 2 > chunk_chars:
            flush()
        buf.append(p)
        buf_len += len(p) + 2

    flush()

    if overlap_chars > 0 and len(chunks) > 1:
        out2: List[str] = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
            out2.append((tail + "\n\n" + cur).strip())
        chunks = out2

    return [c for c in chunks if c]


def chunk_markdown_by_sections(
    markdown: str, *, chunk_chars: int, overlap_chars: int, drop_references: bool
) -> List[Tuple[str, str]]:
    """
    Section-aware chunking for Docling markdown.
    Returns a list of (section_title, chunk_text) where chunk_text includes the section title as a header.
    """
    md = markdown
    if drop_references:
        md = _strip_references_section(md)
    md = _strip_docling_noise(md)

    out: List[Tuple[str, str]] = []
    for title, body in split_markdown_h2_sections(md):
        if not body.strip():
            continue
        if drop_references and title.strip().lower() in {"references", "reference"}:
            continue
        for part in chunk_section_text(body, chunk_chars=chunk_chars, overlap_chars=overlap_chars):
            out.append((title, (f"## {title}\n\n{part}").strip()))
    return out


@dataclass
class ChunkRec:
    paper: str
    chunk_index: int
    section_title: str
    text: str


def embed_texts(client: OpenAI, model: str, texts: Sequence[str]) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=list(texts))
    mat = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
    return _normalize_rows(mat)


def load_or_build_paper_chunk_index(
    *,
    client: OpenAI,
    paper: str,
    corpus_root: Path,
    cache_dir: Path,
    embedding_model: str,
    chunk_chars: int,
    overlap_chars: int,
    batch_size: int,
) -> Tuple[List[ChunkRec], np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    text = load_paper_text(corpus_root, paper)

    section_chunks = chunk_markdown_by_sections(
        text, chunk_chars=chunk_chars, overlap_chars=overlap_chars, drop_references=True
    )
    chunks = [c for (_t, c) in section_chunks]
    section_titles = [t for (t, _c) in section_chunks]
    spec_hash = _sha256(
        "chunking=v2\n"
        + f"{paper}\n{embedding_model}\n{chunk_chars}\n{overlap_chars}\n{len(chunks)}\n"
        + "\n".join(section_titles[:50])
    )
    meta_path = cache_dir / f"{paper}.chunks.json"
    emb_path = cache_dir / f"{paper}.chunks.emb.npy"

    if meta_path.exists() and emb_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("spec_hash") == spec_hash and int(meta.get("num_chunks", -1)) == len(chunks):
                embs = np.load(emb_path).astype(np.float32, copy=False)
                if embs.shape[0] == len(chunks):
                    recs = [
                        ChunkRec(paper=paper, chunk_index=i, section_title=section_titles[i], text=chunks[i])
                        for i in range(len(chunks))
                    ]
                    return recs, _normalize_rows(embs)
        except Exception:
            pass

    # build embeddings
    all_embs: List[np.ndarray] = []
    for i in range(0, len(chunks), max(1, batch_size)):
        batch = chunks[i : i + batch_size]
        all_embs.append(embed_texts(client, embedding_model, batch))
    embs = np.vstack(all_embs) if all_embs else np.zeros((0, 0), dtype=np.float32)

    meta_path.write_text(
        json.dumps(
            {
                "paper": paper,
                "embedding_model": embedding_model,
                "chunk_chars": chunk_chars,
                "overlap_chars": overlap_chars,
                "num_chunks": len(chunks),
                "spec_hash": spec_hash,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    np.save(emb_path, embs)

    recs = [
        ChunkRec(paper=paper, chunk_index=i, section_title=section_titles[i], text=chunks[i])
        for i in range(len(chunks))
    ]
    return recs, embs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", type=Path, default=Path("rag_index"))
    ap.add_argument("--corpus-root", type=Path, default=Path("RAG_Corpus"))
    ap.add_argument("--cache-dir", type=Path, default=Path("rag_text_cache"))

    ap.add_argument("--input-filename", type=str, required=True, help="e.g. input3.png")
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument(
        "--use-image-caption-in-retrieval",
        action="store_true",
        default=True,
        help="Include the query image caption in the chunk-retrieval query embedding (recommended).",
    )
    ap.add_argument("--no-use-image-caption-in-retrieval", dest="use_image_caption_in_retrieval", action="store_false")

    ap.add_argument("--top-k-images", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0, help="Image/text blend for the image retrieval stage.")
    ap.add_argument("--include-non-micrographs", action="store_true", help="Do not filter out non-micrograph results.")

    ap.add_argument("--chunk-chars", type=int, default=2000)
    ap.add_argument("--overlap-chars", type=int, default=300)
    ap.add_argument("--top-n-chunks", type=int, default=12)
    ap.add_argument("--max-papers", type=int, default=10)

    ap.add_argument("--embedding-model", type=str, default="text-embedding-3-large")
    ap.add_argument("--answer-model", type=str, default="gpt-4o")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    client = OpenAI()

    query_meta, top_images = retrieve_top_images(
        index_dir=args.index_dir,
        input_filename=args.input_filename,
        top_k=args.top_k_images,
        alpha=args.alpha,
        only_micrographs=(not args.include_non_micrographs),
    )

    papers: List[str] = []
    for r in top_images:
        p = r.get("paper")
        if isinstance(p, str) and p and p not in papers:
            papers.append(p)
    papers = papers[: args.max_papers]

    # Build chunk indices for selected papers
    all_recs: List[ChunkRec] = []
    all_embs: List[np.ndarray] = []
    for p in papers:
        recs, embs = load_or_build_paper_chunk_index(
            client=client,
            paper=p,
            corpus_root=args.corpus_root,
            cache_dir=args.cache_dir,
            embedding_model=args.embedding_model,
            chunk_chars=args.chunk_chars,
            overlap_chars=args.overlap_chars,
            batch_size=args.batch_size,
        )
        if embs.size == 0:
            continue
        all_recs.extend(recs)
        all_embs.append(embs)

    if not all_embs:
        raise SystemExit("No paper text embeddings available for the selected papers.")

    chunk_embs = np.vstack(all_embs)

    query_caption = str(query_meta.get("caption", "") or "").strip()
    retrieval_context = (
        "Retrieve paper sections relevant to answering a materials-science question about a materials micrograph. "
        "Prioritize synthesis, processing, fabrication, precursors, heat treatment, atmosphere, and experimental methods. "
        "Ignore references lists."
    )
    retrieval_query = f"{retrieval_context}\n\nResearch question:\n{args.question}"
    if args.use_image_caption_in_retrieval and query_caption:
        retrieval_query += f"\n\nQuery image caption:\n{query_caption}"

    # Embed retrieval query (system-style context + question [+ caption])
    q_emb = embed_texts(client, args.embedding_model, [retrieval_query])[0]
    sims = chunk_embs @ q_emb
    top_idx = np.argsort(-sims)[: args.top_n_chunks]

    retrieved = []
    for i in top_idx:
        rec = all_recs[int(i)]
        retrieved.append(
            {
                "rank": len(retrieved) + 1,
                "score": float(sims[int(i)]),
                "paper": rec.paper,
                "chunk_index": rec.chunk_index,
                "section_title": rec.section_title,
                "text": rec.text,
            }
        )

    context_blocks = []
    for r in retrieved:
        context_blocks.append(
            f"[{r['paper']} | {r.get('section_title','')} | chunk {r['chunk_index']} | sim={r['score']:.3f}]\n{r['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are a helpful materials-science assistant. "
        "Answer using only the provided evidence snippets when possible. "
        "When making inferences, clearly label them as hypotheses. "
        "Cite sources inline using the paper IDs from the snippets (e.g., [1-s2.0-...])."
    )

    user = f"""User question:
{args.question}

Query image caption (auto-generated, may be imperfect):
{query_caption or "(no caption available)"}

Evidence snippets from relevant papers:
{context}

Write a concise answer. Include:
- likely synthesis/process hints supported by the snippets
- any key parameters explicitly mentioned (temperature, time, precursors, atmosphere, etc.)
- uncertainty notes when evidence is missing
"""

    resp = client.responses.create(
        model=args.answer_model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    out = {
        "input_filename": args.input_filename,
        "question": args.question,
        "top_images": top_images,
        "papers": papers,
        "retrieved_chunks": retrieved,
        "answer": (resp.output_text or "").strip(),
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


