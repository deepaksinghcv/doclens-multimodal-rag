"""Central configuration (Decision D3). All tunable knobs live here, no magic numbers."""

from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"  # persistent vector store lives here

# --- Chunking (D1) ---
CHUNK_SIZE_WORDS = 250    # ~325 tokens, safely under bge's 512-token limit
CHUNK_OVERLAP_WORDS = 40  # ~15% overlap, cushions procedures split at boundaries

# --- Embeddings (D4) ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge wants an instruction prefix on the QUERY only, never on documents:
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Vector store / retrieval ---
COLLECTION_NAME = "doclens"
TOP_K = 5  # final number of chunks returned to the generator

# --- Reranking (two-stage retrieval) ---
# Stage 1 (bi-encoder) casts a wide net of CANDIDATE_K; stage 2 (cross-encoder) reranks
# them down to TOP_K. CANDIDATE_K must be > TOP_K to give the reranker room to work.
RERANK_MODEL = "BAAI/bge-reranker-base"
CANDIDATE_K = 20

# --- Generation ---
OLLAMA_MODEL = "qwen3:4b"

# --- Multimodal (D6: caption-and-embed) ---
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"
MIN_IMAGE_SIZE = 250  # px; below this is almost always icons/logos, not diagrams
MAX_CAPTIONS_PER_DOC = 40  # cap VLM captioning work per document (local inference is slow)
VLM_MODEL = "qwen2.5vl:3b"  # vision-language model for captioning + image reasoning
