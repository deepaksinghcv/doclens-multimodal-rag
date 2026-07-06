"""
Box 1.5 — CLEAN (between PARSE and CHUNK).

Remove running headers/footers and repeated boilerplate that PyMuPDF faithfully extracts
from every page. Technique: content is unique to a page, boilerplate repeats across pages.
We normalize each line (strip digits so "Manual 45" == "Manual 46"), count how many pages
each normalized line appears on, and drop the ones that repeat too often.

Pipeline position:  parse_pdf -> clean_pages -> chunk_pages
"""

import re
from collections import Counter

# Only short lines are boilerplate candidates; real paragraphs are long and shouldn't be
# stripped even if they somehow repeat.
MAX_BOILERPLATE_LEN = 70
MIN_PAGE_FRACTION = 0.4  # a line on >= 40% of pages is treated as boilerplate


def _normalize(line: str) -> str:
    """Canonical form for repetition matching: drop digits, collapse whitespace, lowercase."""
    no_digits = re.sub(r"\d+", "", line)
    return re.sub(r"\s+", " ", no_digits).strip().lower()


def find_boilerplate(pages: list[dict], min_fraction: float = MIN_PAGE_FRACTION) -> set[str]:
    """Return the set of NORMALIZED lines that repeat across enough pages to be boilerplate.

    A normalized line counts at most ONCE per page (so a line repeated within one page
    doesn't inflate the count).
    """
    num_pages = len(pages)
    if num_pages == 0:
        return set()

    page_counts: Counter[str] = Counter()

    # Count each distinct normalized candidate line once per page.
    for page in pages:
        seen_on_page = {
            norm
            for line in page["text"].splitlines()
            if (norm := _normalize(line)) and len(norm) <= MAX_BOILERPLATE_LEN
        }
        page_counts.update(seen_on_page)

    # Boilerplate = appears on at least `min_fraction` of all pages.
    threshold = min_fraction * num_pages
    return {line for line, count in page_counts.items() if count >= threshold}


def clean_pages(pages: list[dict], min_fraction: float = MIN_PAGE_FRACTION) -> list[dict]:
    """Strip boilerplate lines from every page. Returns new page dicts (same shape)."""
    boiler = find_boilerplate(pages, min_fraction=min_fraction)

    cleaned: list[dict] = []
    for page in pages:
        kept = [
            line for line in page["text"].splitlines() if _normalize(line) not in boiler
        ]
        cleaned.append({"page": page["page"], "text": "\n".join(kept)})

    return cleaned


if __name__ == "__main__":
    from ingestion.parse_pdf import parse_pdf

    pages = parse_pdf("data/pdfs/psr_i500.pdf")
    boiler = find_boilerplate(pages)
    print(f"Detected {len(boiler)} boilerplate lines. Examples:")
    for b in list(boiler)[:10]:
        print(f"  - {b!r}")

    cleaned = clean_pages(pages)
    before = len(pages[44]["text"])
    after = len(cleaned[44]["text"])
    print(f"\nPage 45 chars: {before} -> {after}  (removed {before - after})")
