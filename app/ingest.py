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
from ingestion.caption import caption_pdf_images
from ingestion.chunker import chunk_pages
from ingestion.clean import clean_pages
from ingestion.parse_pdf import parse_pdf
from vectordb.chroma_store import add_chunks


def ingest_pdf(pdf_path: Path, with_images: bool = True) -> int:
    """Parse -> clean -> chunk (+ caption figures) -> store one PDF.

    Text chunks and image-caption chunks go into ONE collection (D6). Returns total indexed.
    """
    source = pdf_path.name  # filename -> stored in metadata for citations

    pages = parse_pdf(pdf_path=pdf_path)
    pages = clean_pages(pages)  # strip repeated headers/footers before chunking
    text_chunks = chunk_pages(pages=pages)

    # Multimodal: extract + caption figures into chunks that share the same index.
    image_chunks = caption_pdf_images(str(pdf_path)) if with_images else []

    # One monotonic chunk_index across text + image chunks (unique ids per source).
    all_chunks = text_chunks + image_chunks
    for idx, chunk in enumerate(all_chunks):
        chunk["chunk_index"] = idx

    return add_chunks(chunks=all_chunks, source=source)

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
