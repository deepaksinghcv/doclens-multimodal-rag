"""
Box 1 — PARSE.

Turn a PDF file into a list of per-page text records. This is the ONLY job of this
module: file -> [{"page": 1, "text": "..."}, ...]. No chunking, no cleaning.

The page number is 1-based (human-facing) because it becomes our citation later.
"""

from pathlib import Path

import fitz  # PyMuPDF imports under the historical name "fitz"


def parse_pdf(pdf_path: str | Path) -> list[dict]:
    """Extract text from every page of a PDF.

    Args:
        pdf_path: path to a .pdf file.

    Returns:
        A list of dicts, one per page, in document order:
            [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
        `page` is 1-based. `text` may be an empty string for blank/scanned pages.
    """

    pages_list = []
    with fitz.open(pdf_path) as doc:
        num_pages = len(doc) #get number of pages
        for i in range(num_pages):
            current_page = doc.load_page(i) #load a page
            page_text = current_page.get_text("text") # type: ignore[attr-defined]
            page_dict = {
                "page": i + 1, #using i+1 since humans prefer 1-indexed numbering
                "text": page_text
            }
            pages_list.append(page_dict)
    
    return pages_list #return list of dictionaries




if __name__ == "__main__":
    # Smoke test: run `python -m ingestion.parse_pdf` after dropping a PDF in data/pdfs/.
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/pdfs/psr_i500.pdf"
    pages = parse_pdf(pdf)
    print(f"Parsed {len(pages)} pages from {pdf}")
    for p in pages[:2]:
        preview = p["text"][:200].replace("\n", " ")
        print(f"  page {p['page']}: {preview!r}")
