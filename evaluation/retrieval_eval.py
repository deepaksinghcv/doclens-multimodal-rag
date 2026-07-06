"""
Phase 4 — Retrieval evaluation.

Scores the retriever against the gold set with two metrics:
  - Recall@K : did the correct (source, page) appear anywhere in the top-K?  (found-at-all)
  - MRR      : 1/rank of the first correct hit, averaged.                     (ranked-well)

Run:  python -m evaluation.retrieval_eval
"""

import json
from pathlib import Path
from config import TOP_K
from retrieval.retrieve import retrieve

EVAL_SET = Path(__file__).parent / "eval_set.json"


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET.read_text())


def first_relevant_rank(hits: list[dict], source: str, pages: list[int]) -> int | None:
    """Return the 1-based rank of the first hit matching (source, page), else None.

    A hit matches when hit["source"] == source AND hit["page"] in pages.
    """
    """
    #single page hits
    for i, hit in enumerate(hits,start=1):
        hit_source = hit["source"]
        hit_page = hit["page"]

        if page == hit_page and source == hit_source:
            return i
    return None
    """

    #multipage hits
    for i, hit in enumerate(hits,start=1):
        hit_source = hit["source"]
        hit_page = hit["page"]

        if hit_page in pages and source == hit_source:
            return i
    return None


def evaluate(retriever, label: str, top_k: int = TOP_K) -> None:
    """Score a retriever function (query, top_k) -> hits against the gold set."""
    data = load_eval_set()
    ranks: list[int | None] = []

    print(f"\n=== {label} @ K={top_k} ===")
    for item in data:
        hits = retriever(item["question"], top_k=top_k)
        rank = first_relevant_rank(hits, item["source"], item["expected_pages"])
        ranks.append(rank)
        status = f"rank {rank}" if rank else "MISS"
        print(f"  [{status:>7}] {item['question']}")

    recall_at_k = sum(1 for rank in ranks if rank is not None) / len(ranks)
    mrr = sum(1 / rank for rank in ranks if rank is not None) / len(ranks)
    print(f"  Recall@{top_k}: {recall_at_k:.3f}   MRR: {mrr:.3f}")


if __name__ == "__main__":
    from retrieval.rerank import retrieve_and_rerank

    evaluate(retrieve, "Baseline (bi-encoder only)")
    evaluate(retrieve_and_rerank, "Reranked (bi-encoder + cross-encoder)")
