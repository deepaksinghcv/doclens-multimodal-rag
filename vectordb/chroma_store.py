"""
Box 4 — STORE.

Persist chunks + their embeddings into a Chroma collection so we can search by similarity.

Each record is four ALIGNED lists (same length, same order):
    ids[i], embeddings[i], documents[i], metadatas[i]  all describe the same chunk.

We pass our OWN embeddings (bge, per D4) instead of letting Chroma embed for us, so we
control the model on both the store and the query side.
"""

import hashlib
import logging
# import os
# os.environ["ANONYMIZED_TELEMETRY"] = "False"
import chromadb
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
from config import CHROMA_DIR, COLLECTION_NAME
from embeddings.embed import embed_documents


# The Chroma client/collection is expensive to construct and cheap to reuse, so we build
# it once per process and cache it (singleton) — not on every query.
_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    """Open (or create) the persistent Chroma collection, configured for cosine space.

    Cached: the PersistentClient is created once per process and reused for every call.
    """
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # match our normalized vectors
        )
    return _collection


def list_sources() -> list[str]:
    """Return the distinct manual filenames currently indexed."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    metadatas = collection.get(include=["metadatas"]).get("metadatas") or []
    return sorted({m["source"] for m in metadatas if m.get("source")})


def _chunk_id(source: str, chunk_index: int) -> str:
    """Deterministic id for a chunk -> re-ingesting the same content upserts, not duplicates."""
    return hashlib.sha1(f"{source}|{chunk_index}".encode()).hexdigest()


def add_chunks(chunks: list[dict], source: str) -> int:
    """Embed chunks and upsert them into the collection.

    Args:
        chunks: output of chunk_pages -> [{"text", "page", "chunk_index"}, ...]
        source: the document name (e.g. "psr_i500.pdf"), stored in metadata for citations.

    Returns:
        The number of chunks written.
    """
    if not chunks:
        return 0

    collection = get_collection()
    collection.delete(where={"source": source})

    
    texts = [c['text'] for c in chunks]
    
    embeddings = embed_documents(texts)
    
    ids = [
        _chunk_id(source, c["chunk_index"])
        for c in chunks
    ]
    documents = texts
    metadatas = [
        {
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "source": source,
            "type": c.get("type", "text"),          # "text" | "image"
            "image_path": c.get("image_path", ""),   # path to figure, for image chunks
        }
        for c in chunks
    ]

    collection.upsert(
        ids = ids,
        documents = documents,
        embeddings = embeddings,
        metadatas = metadatas
    )

    return len(chunks)


if __name__ == "__main__":
    # Smoke test: run the FULL ingestion front-end and store it.
    from ingestion.chunker import chunk_pages
    from ingestion.parse_pdf import parse_pdf

    source = "psr_i500.pdf"
    chunks = chunk_pages(parse_pdf(f"data/pdfs/{source}"))
    n = add_chunks(chunks, source)

    col = get_collection()
    
    print(f"stored {n} chunks; collection now holds {col.count()} records")
