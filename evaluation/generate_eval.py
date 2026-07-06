"""
Grow the retrieval gold set from real section headings (no LLM — fast & reproducible).

Method: harvest section-title headings from each manual and turn each into a topic->section
question; the gold answer is the page the heading lives on (verified by construction). This
gives broad coverage cheaply.

Honesty caveat: heading-derived questions share wording with their page, so retrieval is
*easier* than for natural user questions — treat these metrics as an optimistic upper bound.
The hand-authored questions (kept at the top of eval_set.json) remain the conservative
reference. See README for both.

Run:  python -m evaluation.generate_eval
"""

import json
from pathlib import Path

from ingestion.clean import clean_pages
from ingestion.parse_pdf import parse_pdf

EVAL_SET = Path(__file__).parent / "eval_set.json"

# Generic / navigational headings that don't make good topic questions.
STOP = {
    "contents", "index", "reference", "appendix", "note", "menu", "warning", "caution",
    "introduction", "specifications", "troubleshooting", "table of contents", "setting up",
    "quick guide", "about this guide", "sample photos", "part names", "software overview",
    "basic operation and display items", "handling precautions", "safety instructions",
}

# (source, pdf path, page range, max questions to take)
SPECS = [
    ("psr_i500.pdf", "data/pdfs/psr_i500.pdf", (12, 80), 25),
    ("canon_m50_mark2.pdf", "data/pdfs/canon_m50_mark2.pdf", (12, 590), 61),
]


def harvest_headings(pages: list[dict], lo: int, hi: int) -> list[tuple[int, str]]:
    """Pull title-like section headings (page, heading), deduped, in page order."""
    out: list[tuple[int, str]] = []
    seen: set[str] = set()
    for p in pages:
        if not (lo <= p["page"] <= hi):
            continue
        for line in p["text"].splitlines()[:3]:
            s = " ".join(line.split())
            words = s.split()
            if not (2 <= len(words) <= 6) or len(s) > 45 or s[-1:] in ".,:;":
                continue
            if any(c.isdigit() for c in s) or s.lower() in STOP or s.lower() in seen:
                continue
            caps = sum(1 for w in words if w[:1].isupper())
            if caps >= max(2, len(words) - 1):  # title-like
                seen.add(s.lower())
                out.append((p["page"], s))
                break
    return out


def main() -> None:
    existing = json.loads(EVAL_SET.read_text())
    seen_q = {e["question"].lower() for e in existing}
    new: list[dict] = []

    for source, path, (lo, hi), limit in SPECS:
        headings = harvest_headings(clean_pages(parse_pdf(path)), lo, hi)[:limit]
        for page, heading in headings:
            q = f"Where can I find information about {heading}?"
            if q.lower() in seen_q:
                continue
            seen_q.add(q.lower())
            new.append({"question": q, "source": source, "expected_pages": [page]})
        print(f"{source}: added {sum(1 for e in new if e['source'] == source)} questions")

    combined = existing + new
    EVAL_SET.write_text(json.dumps(combined, indent=2) + "\n")
    print(f"Total now: {len(combined)} questions")


if __name__ == "__main__":
    main()
