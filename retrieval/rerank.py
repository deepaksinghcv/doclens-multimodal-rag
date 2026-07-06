"""
Stage 2 — RERANK (cross-encoder).

Two-stage retrieval: the bi-encoder (retrieve) casts a wide net of CANDIDATE_K chunks; this
cross-encoder re-scores each (query, chunk) PAIR jointly and reorders them, so the truly
relevant chunk rises to the top. Reranking only REORDERS the candidates — it cannot recover
a chunk stage 1 never retrieved.
"""

from sentence_transformers import CrossEncoder

from config import CANDIDATE_K, RERANK_MODEL, TOP_K
from retrieval.retrieve import retrieve

# Heavy model -> load once (lazy singleton), same pattern as the embedder.
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank(query: str, hits: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Re-score candidate hits with the cross-encoder and return the top_k.

    Args:
        query: the user's question.
        hits:  candidate chunks from stage 1 (each has a "text" key).
        top_k: how many to return after reranking.

    Returns:
        The top_k hits, sorted by cross-encoder relevance (desc), each with a new
        "rerank_score" key added.
    """
    if not hits:
        return []

    # TODO 1: Build the (query, chunk_text) PAIRS the cross-encoder scores:
    #   pairs = [(query, h["text"]) for h in hits]
    pairs = [(query, hit["text"]) for hit in hits]

    # TODO 2: Score them all at once:
    #   scores = get_reranker().predict(pairs)   # one float per pair; higher = more relevant
    scores = get_reranker().predict(pairs)
    # TODO 3: Attach each score to its hit:
    #   for h, s in zip(hits, scores): h["rerank_score"] = float(s)
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)

    # TODO 4: Sort hits by "rerank_score" descending and return the first top_k:
    #   return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_k]
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_k]
    


def retrieve_and_rerank(
    query: str, top_k: int = TOP_K, candidate_k: int = CANDIDATE_K, source: str | None = None
) -> list[dict]:
    """Full two-stage retrieval: wide bi-encoder net -> cross-encoder reorder -> top_k."""
    candidates = retrieve(query, top_k=candidate_k, source=source)  # stage 1: recall
    return rerank(query, candidates, top_k=top_k)                    # stage 2: precision


if __name__ == "__main__":
    # See the reorder in action: compare stage-1 order vs reranked order.
    q = "How do I connect the camera to a PC?"

    stage1 = retrieve(q, top_k=CANDIDATE_K)
    reranked = retrieve_and_rerank(q)

    print(f"Q: {q}\n")
    print("Stage 1 (bi-encoder) top 5 pages:", [h["page"] for h in stage1[:5]])
    print("After rerank      top 5 pages:", [h["page"] for h in reranked])
    print(f"\nTop reranked chunk (page {reranked[0]['page']}):")
    print(" ", reranked[0]["text"][:160].replace("\n", " "))
