"""
Box 6 — RETRIEVE.

Given a question, return the top-K most similar chunks (with their page metadata) from the
Chroma index. This is the bridge from a user's words to the evidence the LLM will read.

Reuses embed_query (same bge model + D4 query prefix) and the existing collection.
"""

from config import TOP_K
from embeddings.embed import embed_query
from vectordb.chroma_store import get_collection


def retrieve(query: str, top_k: int = TOP_K, source: str | None = None) -> list[dict]:
    """Find the top_k chunks most semantically similar to `query`.

    Args:
        query: the user's natural-language question.
        top_k: how many chunks to return.
        source: if given, restrict retrieval to chunks from this manual (metadata filter).

    Returns:
        A list of up to top_k dicts, most-similar first:
            [{"text": str, "page": int, "chunk_index": int, "score": float}, ...]
        score = 1 - cosine_distance, so higher = more similar.
    """

    query_embed = embed_query(query)
    results = get_collection().query(
        query_embeddings=[query_embed],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where={"source": source} if source else None,  # scope to one manual if requested
    )

    #since we are querying only with 1 query, we index the first element from the results
    docs        = results["documents"][0]
    metadatas   = results["metadatas"][0]
    distances   = results["distances"][0]

    return_list = [
        {
            "text": doc,
            "page": meta["page"],
            "chunk_index": meta["chunk_index"],
            "score": 1 - dist,
            "source": meta["source"],
            "type": meta.get("type", "text"),          # "text" | "image"
            "image_path": meta.get("image_path", ""),   # figure path for image chunks
        }
        for doc, meta, dist in zip(docs, metadatas, distances)
    ]

    return return_list

if __name__ == "__main__":
    # Smoke test: ask a real question against the indexed manual.
    question = "How do I connect headphones?"
    hits = retrieve(question)
    print(f"Q: {question}\n")
    for rank, h in enumerate(hits, 1):
        preview = h["text"][:120].replace("\n", " ")
        print(f"#{rank}  page {h['page']}  score={h['score']:.3f}")
        print(f"     {preview!r}\n")
