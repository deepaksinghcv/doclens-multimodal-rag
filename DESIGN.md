# DocLens — Design Decisions

Locked decisions for the build. Each records the *choice* and the *why*, because the
reasoning is the part worth defending.

## D1 — Chunking: page-aware, ~250 words / ~15% overlap (≈325 tokens)
Every chunk carries its source `page` in metadata — this is our citation mechanism, so we
chunk *within* a page (a chunk never spans a page boundary). Skeleton uses a simple
sliding word-window with overlap; we upgrade to structure-aware recursive splitting
(paragraphs → sentences) in the improvement phase (per D5).
**Why size:** bge-small-en-v1.5 has a hard **512-token** input limit — anything longer is
*silently truncated* before embedding. ~250 words ≈ ~325 tokens leaves safe headroom. (An
earlier draft said "700 tokens", which would have been truncated — corrected.)
**Why overlap:** technical manuals are procedural; window boundaries split a procedure
mid-step. Overlap is cheap insurance against boundary loss.

## D2 — Citations are programmatic, not model-generated
The page number attached to an answer comes from the *retrieved chunk's metadata*, in code.
The LLM writes prose; the pipeline owns the citation.
**Why:** trusting the model to report its source is a hallucination vector and would
invalidate our own faithfulness / citation-correctness metrics.

## D3 — Central config.py
Model names, chunk size, overlap, top_k, paths live in one module. No magic numbers.

## D4 — BGE asymmetric embedding (query-only instruction prefix)
bge-small-en-v1.5 expects an instruction prefix on the *query* only, not on documents.
**Why:** mismatching query/document encoding silently degrades recall.

## D5 — Walking skeleton before polishing components
Build a thin end-to-end slice (parse → chunk → embed → store → retrieve → answer) first,
then improve each stage against the eval set.
**Why:** avoids perfecting a parser for a pipeline that doesn't exist yet.

## D6 — Multimodal (Phase 3): caption-and-embed, not joint CLIP
At ingestion, a vision model captions each diagram; the caption text is embedded into the
single unified text index. At answer time the actual image is passed to a VLM.
**Why:** one text index (simpler retrieval + eval); VLM captions of technical diagrams
retrieve better than raw CLIP image vectors.

## D7 — LoRA comparison dropped to stretch goal
RAG quality is dominated by retrieval, not generator weights. Higher-payoff stretch goals:
cross-encoder reranker, hybrid BM25+dense, query rewriting.

## Metadata schema (per chunk)
`doc_id`, `source` (filename), `page`, `chunk_index`, content-hash id (stable, idempotent).

## Orchestration
One `app/ingest.py` entry point (parse → chunk → embed → store) and one `app/cli.py` for
querying. Components stay importable modules; scripts just wire them.
