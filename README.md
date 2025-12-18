### VLRIMM — Visual Literature Retrieval for SEM Micrographs (Multi‑Modal RAG)

As the “AI for Science” movement accelerates, the volume of scientific literature and high‑resolution characterization data continues to grow rapidly. While Large Language Models (LLMs) offer powerful reasoning, they have two critical limitations in a lab setting:

- **Knowledge cutoff**: they cannot inherently access the latest literature.
- **Hallucinations**: they may invent domain‑specific details when evidence is missing.

In materials science, the “ground truth” of sample morphology is often captured with **Scanning Electron Microscopy (SEM)** (and related micrographs such as TEM, STEM, EBSD/IPF maps, etc.). Papers frequently provide limited or biased textual description of morphology, making it hard to search the literature using what matters most: the microstructure itself.

**VLRIMM** addresses this with a **multi‑modal Retrieval‑Augmented Generation (RAG)** pipeline. It enables researchers to use a **micrograph as the query** and retrieve **scientific papers with similar micrographs**, combining:

- **Image embeddings** (DINOv3 / vision foundation model features)
- **Text embeddings** (OpenAI embeddings over VLM‑generated captions)

The retrieved evidence can optionally be used to generate a **grounded answer** to a user question by running text‑RAG over the papers associated with the top retrieved images.

---

### What’s in this repo

Core scripts (the “demo pipeline”):

- **`pdf_processor.py`**: Extracts paper text + figure crops into `docling_output/` (resumable via `.processed` marker).
- **`vlm_caption.py`**: Two‑stage vision pipeline:
  - classify image → **micrograph vs not**
  - caption **only** micrographs
  - optionally archives non‑micrographs while preserving per‑paper structure
  - logs results as resumable JSONL
- **`rag_prepare_corpus.py`**: Builds a unified, portable corpus folder `RAG_Corpus/` (copies or symlinks).
- **`rag_build_index.py`**: Builds `rag_index/` (`*_meta.jsonl` + `*_image_embeddings.npy`) from:
  - DINOv3 image embeddings (pickle)
  - VLM caption JSONL outputs
- **`rag_embed_text.py`**: Computes OpenAI text embeddings for captions (resumable cache JSONL).
- **`rag_query.py`**: Retrieves top‑K literature images for an input image (image‑only or hybrid image+text).
- **`rag_answer.py`**: “Textual RAG” stage: chunk + embed only the relevant papers, retrieve best chunks, then generate a cited answer.
- **`rag_alpha_sweep_export.py`**: Exports “Query + Top‑K” grids across multiple \(\alpha\) values for quick demos.

Notebook:

- **`rag_demo.ipynb`**: End‑to‑end demo (retrieval visualization + optional text‑RAG answer).

Optional (only needed if you want to *compute* DINOv3 embeddings yourself):

- **`image_encoder/`**: helper scripts for extracting DINOv3 features (requires `torch`, `transformers`, etc.).

---

### Data layout (generated artifacts)

This repo is designed to keep the **code** public while allowing **data** to be generated locally.

Typical generated folders:

- **`Papers/`**: input PDFs (not meant to be committed)
- **`docling_output/<paper>/`**: extracted markdown + figures (generated)
- **`RAG_Corpus/`**:
  - `Inputs/` (query images)
  - `Literature/<paper>/images_processed/` (paper figures)
  - `Literature/<paper>/<paper>.txt` (paper text)
- **`rag_index/`**: numpy matrices + jsonl metadata for retrieval
- **`rag_text_cache/`**: cached chunk embeddings for `rag_answer.py`
- **`archive_non_micrographs/`**: archived images filtered out by `vlm_caption.py` (optional)

---

### Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you plan to run any OpenAI steps, export an API key:

```bash
export OPENAI_API_KEY="..."
```

---

### Pipeline (recommended order)

#### 1) Extract papers (PDF → text + images)

```bash
python pdf_processor.py --input-dir Papers --output-dir docling_output
```

#### 2) Caption paper figures (micrograph filter → caption only micrographs)

```bash
python vlm_caption.py \
  --images-root docling_output \
  --results-file vlm_results.jsonl \
  --archive-root archive_non_micrographs
```

#### 3) Caption input/query images

```bash
python vlm_caption.py \
  --images-root input_images \
  --results-file input_vlm_results.jsonl \
  --no-require-images-processed
```

#### 4) Build a unified corpus folder (portable file paths)

```bash
python rag_prepare_corpus.py \
  --docling-root docling_output \
  --input-images-dir input_images \
  --corpus-root RAG_Corpus \
  --mode copy \
  --overwrite
```

#### 5) Build the retrieval index (NO API calls)

This step expects you already have image embeddings (e.g. DINOv3) stored in pickle files:

```bash
python rag_build_index.py --out-dir rag_index --corpus-root RAG_Corpus
```

#### 6) Embed captions for hybrid retrieval (OpenAI embeddings)

```bash
python rag_embed_text.py --meta rag_index/literature_meta.jsonl --out rag_index/literature_text_embeddings.npy
python rag_embed_text.py --meta rag_index/input_meta.jsonl      --out rag_index/input_text_embeddings.npy
```

---

### Image retrieval (micrograph → similar literature figures)

Run retrieval for an input image:

```bash
python rag_query.py --input-filename input1.png --top-k 10 --alpha 0.8
```

#### The \(\alpha\) parameter

When caption embeddings are available, scores are blended as:

\[
\text{score} = \alpha \cdot \text{sim}_{img} + (1 - \alpha) \cdot \text{sim}_{txt}
\]

- **\(\alpha = 1.0\)**: image‑only retrieval
- **\(\alpha = 0.0\)**: caption‑only retrieval
- **\(\alpha \in (0,1)\)**: hybrid retrieval

---

### Textual RAG (answer a question grounded in retrieved papers)

Given a query micrograph + a user question, this stage:

- retrieves top‑K similar images
- collects their paper IDs
- chunks + embeds **only those papers**
- retrieves top evidence chunks
- generates an answer **with inline citations**

Example:

```bash
python rag_answer.py \
  --input-filename input1.png \
  --question "Based on this image, describe potential synthesis details of this material" \
  --top-k-images 10 \
  --alpha 0.8
```

---

### Demo exports (alpha sweep grids)

Generate grids like “Query + Top‑10” across multiple \(\alpha\) values:

```bash
python rag_alpha_sweep_export.py --index-dir rag_index --corpus-root RAG_Corpus --out-dir rag_alpha_exports
```

---

### Notes / caveats

- **Costs & latency**: OpenAI vision + embedding calls incur API costs and may take time for large corpora. All OpenAI‑calling scripts are designed to be **resumable**.
- **Data privacy**: ensure PDFs and images you process are permitted for use with external APIs.
- **Reproducibility**: exact results depend on embedding versions/models and any preprocessing.


