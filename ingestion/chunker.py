"""
Box 2 — CHUNK.

Turn parsed pages into retrieval-sized chunks. Each chunk stays within a single page
(so it maps to exactly one citation) and carries enough context to be self-contained.

Skeleton strategy (D1/D5): a simple sliding word-window with overlap. We upgrade to
structure-aware splitting later, once we can measure retrieval quality.

    parse_pdf output ->  [{"page": 1, "text": "..."}, ...]
    chunk_pages output -> [{"text": "...", "page": 1, "chunk_index": 0}, ...]
"""

from config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS


def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[dict]:
    """Split each page's text into overlapping word-windows.

    Args:
        pages: output of parse_pdf -> [{"page": int, "text": str}, ...]
        chunk_size: max words per chunk.
        overlap: words shared between consecutive chunks (must be < chunk_size).

    Returns:
        [{"text": str, "page": int, "chunk_index": int}, ...]
        chunk_index is a single monotonic counter across the whole document (0, 1, 2, ...).
        Empty/whitespace-only pages produce no chunks.
    """
    # Guard: if overlap >= chunk_size, the window never advances -> infinite loop.
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")

    chunks: list[dict] = []
    chunk_index = 0  # monotonic across the whole document

    for page in pages:
        page_index = page['page']
        words = page["text"].split()
        n_words = len(words)

        if n_words > 0: #process only if words are there in the page
            start = 0
            while start < n_words:
                window = words[start : start + chunk_size]
                chunks.append({
                    "chunk_index": chunk_index,
                    "text": " ".join(window),
                    "page": page_index
                })
                start = start + chunk_size - overlap
                chunk_index += 1

    return chunks


if __name__ == "__main__":
    # Smoke test: parse the sample PDF, chunk it, and inspect the result.
    from ingestion.parse_pdf import parse_pdf

    pages = parse_pdf("data/pdfs/psr_i500.pdf")
    chunks = chunk_pages(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    if chunks:
        # print(chunks[0])
        # print("------")
        # print(chunks[1])
        # print("-------")
        # print(chunks[2])
        c = chunks[0]
        print(f"  first chunk: page={c['page']} idx={c['chunk_index']} "
              f"words={len(c['text'].split())}")
        print(f"  preview: {c['text'][:160]!r}")

        c2 = chunks[1]
        print(f"  second chunk: page={c2['page']} idx={c2['chunk_index']} "
              f"words={len(c2['text'].split())}")
        print(f"  preview: {c2['text'][:160]!r}")
