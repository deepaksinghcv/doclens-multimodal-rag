"""
app/ingest.py — ingestion entry point.

Orchestrates the offline axis: PDF file(s) -> parsed pages -> chunks -> indexed in Chroma.
Thin wiring only; all real work lives in the ingestion/ and vectordb/ modules.

Usage:
    python -m app.ingest                  # ingest every *.pdf in data/pdfs/
    python -m app.ingest psr_i500.pdf     # ingest one file (name or path)
"""

import sys
from pathlib import Path

from config import PDF_DIR
from ingestion.chunker import chunk_pages
from ingestion.clean import clean_pages
from ingestion.parse_pdf import parse_pdf
from vectordb.chroma_store import add_chunks


def ingest_pdf(pdf_path: Path) -> int:
    """Parse -> clean -> chunk -> store one PDF. Returns the number of chunks indexed."""
    source = pdf_path.name  # filename, e.g. "psr_i500.pdf" -> stored in metadata for citations

    pages = parse_pdf(pdf_path=pdf_path)
    pages = clean_pages(pages)  # strip repeated headers/footers before chunking
    chunks = chunk_pages(pages=pages)
    n = add_chunks(chunks=chunks, source=source)
    return n

def _resolve_targets(args: list[str]) -> list[Path]:
    """Figure out which PDFs to ingest from CLI args (or all of them if none given)."""
    if args:
        # A name like "psr_i500.pdf" resolves under PDF_DIR; a full path is used as-is.
        return [PDF_DIR / a if not Path(a).is_absolute() else Path(a) for a in args]
    return sorted(PDF_DIR.glob("*.pdf"))


def main() -> None:
    targets = _resolve_targets(sys.argv[1:])
    if not targets:
        print(f"No PDFs found in {PDF_DIR}. Drop one in and try again.")
        return

    total = 0
    for pdf in targets:
        if not pdf.exists():
            print(f"  skip (not found): {pdf}")
            continue
        n = ingest_pdf(pdf)
        total += n
        print(f"  ingested {pdf.name}: {n} chunks")
    print(f"Done. {total} chunks across {len(targets)} file(s).")


if __name__ == "__main__":
    main()
