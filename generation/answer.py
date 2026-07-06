"""
Boxes 7+8 — AUGMENT + GENERATE.

Take a question, retrieve evidence, and have a local LLM (Qwen via Ollama) write a grounded
answer. Citations come from the retrieved chunks' metadata (D2), never from the model.
"""

import re
from pathlib import Path

import ollama

from config import OLLAMA_MODEL, TOP_K, VLM_MODEL
from retrieval.retrieve import retrieve


def _strip_think(text: str) -> str:
    """qwen3 may emit <think>...</think> reasoning before its answer. Remove it."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as labeled evidence blocks the model can ground on."""
    return "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in chunks)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """Build the chat messages (system + user) for a grounded, refuse-when-unsure answer.

    Returns a list like:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Requirements for your prompt:
      - SYSTEM: tell the model it is a technical-manual assistant that answers using ONLY
        the provided context, and must say it couldn't find the answer in the manual if the
        context doesn't contain it. Tell it to be concise and not invent steps.
      - USER: include the formatted context (use _format_context(chunks)) AND the question,
        clearly delimited (e.g. a "Context:" block then "Question:" then "Answer:").
    """
    context = _format_context(chunks)
    # TODO: build and return the [system, user] messages list using `context` and `question`.
    message_list = [
        {
            "role"      : "system",
            "content"   : "You are a technical-manual assistant that answers using ONLY the context provided to you, if the answer is not in the manual and context provided to you, DON'T HALLUCINATE OR MAKE UP THE ANSWER, just say that 'couldn't find the answer in the manual(s)'. Also be concise in your answers and dont invent steps. Keep it grounded to the context and content of the manual",
        },
        {
            "role"      : "user",
            "content"   : f"Context:\n{context}\n\nQuestion:{question}\n\nAnswer:"
        }
    ]

    return message_list


def answer(question: str, top_k: int = TOP_K) -> dict:
    """Full RAG: retrieve -> augment -> generate. Returns answer + programmatic citations.

    Multimodal (D6): if any retrieved chunk is an image caption, we route generation to the
    vision model and attach the actual figure(s), so it reasons over pixels — not just the
    caption text.
    """
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {"answer": "No documents are indexed yet.", "pages": [], "chunks": []}

    messages = build_messages(question, chunks)

    # Collect actual figure files for any retrieved image chunks (that still exist on disk).
    image_paths = [
        c["image_path"]
        for c in chunks
        if c.get("type") == "image" and c.get("image_path") and Path(c["image_path"]).exists()
    ]

    if image_paths:
        model = VLM_MODEL
        messages[-1]["images"] = image_paths[:3]  # attach figures to the user turn
    else:
        model = OLLAMA_MODEL

    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0},  # factual grounding, not creativity
    )
    text = _strip_think(response["message"]["content"])

    # Citations come from retrieval metadata (D2), de-duplicated and sorted.
    pages = sorted({c["page"] for c in chunks})

    return {"answer": text, "pages": pages, "chunks": chunks, "used_images": image_paths[:3]}


if __name__ == "__main__":
    q = "How do I connect headphones?"
    result = answer(q)
    print(f"Q: {q}\n")
    print(result["answer"])
    print(f"\nSources: pages {result['pages']}")
