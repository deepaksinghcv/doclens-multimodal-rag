"""
Box (multimodal) — EXTRACT IMAGES.

Pull real figures/diagrams out of a PDF and save them to disk, remembering which page each
came from. Only extraction here — captioning and embedding happen in later steps.

    extract_images("data/pdfs/x.pdf", out_dir)
      -> [{"page": 51, "source": "x.pdf", "image_path": ".../x_p51_x123.png"}, ...]

PyMuPDF image APIs you'll use:
  page.get_images(full=True) -> list of tuples; tuple[0] is the image's `xref` (int id).
  doc.extract_image(xref)    -> dict with "image" (raw bytes), "ext" (e.g. "png"),
                                "width", "height".
"""

from pathlib import Path

import fitz

from config import MIN_IMAGE_SIZE


def extract_images(
    pdf_path: str | Path, out_dir: str | Path, min_size: int = MIN_IMAGE_SIZE
) -> list[dict]:
    """Extract diagram-sized images from a PDF, one file each, tagged with their page.

    Args:
        pdf_path: the PDF to extract from.
        out_dir:  directory to write image files into (created if missing).
        min_size: skip images smaller than this (px) on either dimension.

    Returns:
        [{"page": int, "source": str, "image_path": str}, ...]
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = pdf_path.name

    results: list[dict] = []
    seen_xrefs: set[int] = set()  # dedup: extract each unique image only once

    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            page_num = i + 1

            for img in page.get_images(full=True):
                xref = img[0]
                if xref in seen_xrefs:  # dedup: same logo/diagram referenced many times
                    continue
                seen_xrefs.add(xref)

                info = doc.extract_image(xref)
                if info["width"] < min_size or info["height"] < min_size:
                    continue  # skip icons / logos / tiny fragments

                filename = f"{pdf_path.stem}_p{page_num}_x{xref}.{info['ext']}"
                path = out_dir / filename
                path.write_bytes(info["image"])
                results.append(
                    {"page": page_num, "source": source, "image_path": str(path)}
                )

    return results


if __name__ == "__main__":
    from config import EXTRACTED_IMAGES_DIR

    imgs = extract_images("data/pdfs/canon_m50_mark2.pdf", EXTRACTED_IMAGES_DIR)
    print(f"Extracted {len(imgs)} images")
    for im in imgs[:5]:
        print(f"  page {im['page']}: {im['image_path']}")
