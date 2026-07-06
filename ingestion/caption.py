"""
Box (multimodal) — CAPTION.

Turn extracted figures into searchable text (D6: caption-and-embed). A vision-language
model (Qwen-VL via Ollama) describes each diagram; that caption becomes a chunk that lives
in the SAME text index, tagged type="image" with the path back to the original figure.

    caption_pdf_images("data/pdfs/x.pdf")
      -> [{"text": <caption>, "page": 51, "type": "image", "image_path": "..."}, ...]
"""

import ollama

from config import EXTRACTED_IMAGES_DIR, MAX_CAPTIONS_PER_DOC, VLM_MODEL
from ingestion.extract_images import extract_images

CAPTION_PROMPT = (
    "This is a figure from a technical product manual. In 1-2 sentences, describe what it "
    "shows as specifically as possible: components, buttons, ports, connections, or steps "
    "depicted. Be factual; do not guess beyond what is visible."
)


def caption_image(image_path: str) -> str:
    """Generate a one/two-sentence factual caption for a single image via the VLM."""
    response = ollama.chat(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": CAPTION_PROMPT, "images": [image_path]}],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()


def caption_pdf_images(
    pdf_path: str, limit: int = MAX_CAPTIONS_PER_DOC
) -> list[dict]:
    """Extract a PDF's figures and caption them into image-chunks for the index."""
    images = extract_images(pdf_path, EXTRACTED_IMAGES_DIR)[:limit]

    chunks: list[dict] = []
    for im in images:
        caption = caption_image(im["image_path"])
        chunks.append(
            {
                "text": caption,
                "page": im["page"],
                "type": "image",
                "image_path": im["image_path"],
            }
        )
    return chunks


if __name__ == "__main__":
    caps = caption_pdf_images("data/pdfs/canon_m50_mark2.pdf", limit=3)
    for c in caps:
        print(f"page {c['page']}: {c['text']}\n  ({c['image_path']})\n")
