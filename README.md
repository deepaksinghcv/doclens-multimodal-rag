# DocLens: Local Multimodal RAG for Technical Documents

> Ask questions about manuals, diagrams, and technical PDFs using a fully local, evaluated Retrieval-Augmented Generation (RAG) pipeline — grounded answers with page citations, running entirely on your machine.

📊 **[View the technical deep-dive presentation →](https://doclens-deck.vercel.app)**

---

## Overview

DocLens answers natural-language questions about technical documents (product manuals, engineering guides) by retrieving the relevant passages **and figures** from a PDF and generating a grounded answer with page citations. Everything runs locally — embeddings, vector search, and generation — with no external APIs.

```text
PDF: psr_i500.pdf
Q:   How do I connect headphones?

A:   Connect a pair of headphones to the [PHONES/OUTPUT] jack on the rear panel.
     The built-in speakers are automatically shut off when a plug is inserted.
     Source: page 45
```

Unlike a generic chatbot, DocLens grounds every response in retrieved evidence and refuses when the answer isn't in the document.

---

## ⭐ Highlights

`Recall@5 ≈ 0.93 (reranked)`  ·  `MRR ≈ 0.74`  ·  `100-question gold set`  ·  `100% local`

A summary of the improvements and how each was earned:

| Improvement lever | Problem it solved | Outcome |
|---|---|---|
| **Evaluation harness** (Recall@K, MRR) | "Is retrieval actually good?" | measured on a 100-question gold set; caught its own gold-label blind spots |
| **Boilerplate removal + page-aware chunking** | header/footer noise polluting chunks & citations | clean, correctly page-cited retrieval |
| **Two-stage retrieval** (bi- + cross-encoder), **A/B-tested** | ranking quality | reranker lifts Recall@5 **≈0.86 → ≈0.93** on the broad set (was neutral on 21 hand-Qs — breadth changed the call) |
| **Caption-and-embed multimodal** | figures/diagrams were unsearchable | figures retrievable; VLM reads the actual image to answer |
| **Grounded generation + programmatic citations** | hallucination / untraceable answers | refuses when unsupported; every answer cites its source page |

> Metrics: 100-question gold set — 21 hand-authored (answer pages verified against the manuals) + 79 derived from section headings (higher wording overlap → optimistic). ~±3–5% run-to-run variance. Directional, built to *drive* decisions rather than be statistically conclusive.

---

## Features

- **PDF ingestion** with text extraction (PyMuPDF)
- **Boilerplate removal** — repetition-based header/footer stripping across pages
- **Page-aware chunking** with overlap (citations ride along in metadata)
- **Dense retrieval** via BGE embeddings in a persistent ChromaDB store (idempotent re-indexing)
- **Two-stage retrieval** — bi-encoder recall + optional cross-encoder reranking
- **Grounded generation** — local Qwen3 (Ollama) with a refuse-when-unsupported prompt
- **Programmatic page citations** (computed from retrieval metadata, not the model)
- **Multimodal (caption-and-embed)** — figures are extracted, captioned by a vision-language model (Qwen-VL), embedded into the *same* index, and the actual image is passed to the VLM at answer time
- **Evaluation harness** — Recall@K and MRR over a gold Q→page set

---

## Architecture

A RAG system runs on two time axes — ingestion (offline, once per document) and query (online, per question):

```text
INGESTION (offline)
  PDF ─► parse ─► clean (strip boilerplate) ─► chunk ─┐
      └─► extract images ─► caption (VLM) ────────────┤
                                                       ▼
                                          BGE embeddings ─► ChromaDB
                                          (text + image-caption chunks, one index)

QUERY (online)
  question ─► embed ─► retrieve top-K ─► [optional] cross-encoder rerank
                                            │
                          if an image chunk is retrieved ─► attach figure to VLM
                                            ▼
                          grounded answer + page citations
```

**Key design decisions** are documented in [DESIGN.md](DESIGN.md) — e.g. why citations are programmatic (not model-generated), why we caption-and-embed instead of joint CLIP embeddings, and the bi-encoder/cross-encoder tradeoff.

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| PDF parsing / image extraction | PyMuPDF |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim) |
| Reranker | BAAI/bge-reranker-base (cross-encoder) |
| Vector DB | ChromaDB (persistent, cosine) |
| Generation | Qwen3 (`qwen3:4b`) via Ollama |
| Vision (captions + image QA) | Qwen2.5-VL (`qwen2.5vl:3b`) via Ollama |
| Language | Python 3.12 |

---

## Evaluation

A retrieval harness ([evaluation/retrieval_eval.py](evaluation/retrieval_eval.py)) scores the retriever against a gold set of questions labeled with acceptable source pages:

- **Recall@K** — is a correct page in the top-K? (found-at-all)
- **MRR** — reciprocal rank of the first correct hit (ranked-well)

On a **100-question gold set** (21 hand-authored with manually-verified pages + 79 auto-derived from section headings, see [generate_eval.py](evaluation/generate_eval.py)):

| Retriever | Recall@5 | MRR |
|-----------|----------|-----|
| Bi-encoder (baseline) | ≈ 0.83–0.89 | ≈ 0.69–0.74 |
| + Cross-encoder rerank | ≈ 0.93–0.94 | ≈ 0.74–0.75 |

Two honest caveats: metrics show **~±3–5% run-to-run variance** (borderline ANN rankings flip), and heading-derived questions overlap their page's wording, so treat them as an optimistic upper bound.

The interesting result: on this broad set the **cross-encoder reranker gives a clear recall lift** (≈0.86 → ≈0.93) — the *opposite* of what a smaller 21-question set showed (where it was neutral). A good reminder that **the size and breadth of your eval set can change your conclusions** — which is exactly why the harness exists.

---

## Project Structure

```text
doclens/
├── config.py                 # central config (models, chunk sizes, top_k, paths)
├── ingestion/
│   ├── parse_pdf.py          # PDF -> per-page text
│   ├── clean.py              # cross-page boilerplate removal
│   ├── chunker.py            # page-aware overlapping chunks
│   ├── extract_images.py     # PDF -> figure files
│   └── caption.py            # figures -> VLM captions (image chunks)
├── embeddings/embed.py       # BGE embeddings (query/doc asymmetry)
├── vectordb/chroma_store.py  # persistent Chroma, idempotent upsert
├── retrieval/
│   ├── retrieve.py           # dense bi-encoder search
│   └── rerank.py             # cross-encoder two-stage rerank
├── generation/answer.py      # grounded answer + multimodal routing
├── evaluation/
│   ├── eval_set.json         # gold questions -> source + pages
│   └── retrieval_eval.py     # Recall@K, MRR
└── app/
    ├── ingest.py             # ingest one/all PDFs
    └── cli.py                # interactive Q&A
```

---

## Quickstart

Requires Python 3.12 and [Ollama](https://ollama.com).

```bash
# 1. Environment
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Local models
ollama serve                 # in a separate terminal
ollama pull qwen3:4b         # generator
ollama pull qwen2.5vl:3b     # vision model (for multimodal / captions)

# 3. Add a PDF (source manuals are not committed — bring your own)
cp your_manual.pdf data/pdfs/

# 4. Ingest (parse -> clean -> chunk -> caption figures -> embed -> store)
python -m app.ingest

# 5. Ask questions
python -m app.cli

# Optional: evaluate retrieval
python -m evaluation.retrieval_eval
```

> **Note on documents:** the manuals used during development are copyrighted and are **not** included in this repo. Bring your own PDF and update `evaluation/eval_set.json` with your own gold questions.

---

## Key Learnings

Built to explore practical, defensible RAG engineering:

- Retrieval-augmented generation and embedding-based semantic search
- Bi-encoder vs cross-encoder retrieval (two-stage retrieve-then-rerank)
- Grounded, citation-aware generation and hallucination mitigation
- Multimodal RAG via caption-and-embed
- **Evaluation-driven development** — measuring Recall@K / MRR and making ship/no-ship decisions from data rather than intuition
- Local-first, privacy-preserving deployment
