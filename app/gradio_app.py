"""
app/gradio_app.py — interactive web UI for DocLens.

A thin presentation layer over the pipeline:
  - Ask a question (optionally scoped to one manual) -> grounded answer, source-aware
    citations, and the figures the vision model actually read.
  - Add a new manual (upload a PDF) -> it's parsed, cleaned, chunked, its figures captioned,
    embedded, and indexed live — then immediately queryable.

Run:  python -m app.gradio_app   (opens http://127.0.0.1:7860)
"""

import shutil
from pathlib import Path

import gradio as gr

from app.ingest import ingest_pdf
from config import PDF_DIR
from generation.answer import answer
from vectordb.chroma_store import list_sources

ALL = "All manuals"


def _source_choices() -> list[str]:
    """Dropdown choices: 'All manuals' plus every indexed manual."""
    return [ALL, *list_sources()]


def _indexed_md() -> str:
    sources = list_sources()
    if not sources:
        return "_No manuals indexed yet — add one below._"
    return "**Indexed manuals**\n" + "\n".join(f"- `{s}`" for s in sources)


def ask(question: str, source_choice: str):
    """question (+ optional manual scope) -> (answer_md, sources_md, figure_paths)."""
    if not question.strip():
        return "Ask a question about the indexed manuals.", "", []

    source = None if source_choice in (None, ALL) else source_choice
    result = answer(question, source=source)

    # Source-aware citations: "source · p.N", de-duplicated, order preserved.
    tags: list[str] = []
    for c in result["chunks"]:
        tag = f"`{c['source']}` · p.{c['page']}"
        if tag not in tags:
            tags.append(tag)
    sources = "**Sources**\n" + "\n".join(f"- {t}" for t in tags) if tags else "_No sources._"

    return result["answer"], sources, result.get("used_images", [])


def add_manual(file_path: str, include_figures: bool):
    """Ingest an uploaded PDF, then refresh the manual list + scope dropdown."""
    if not file_path:
        return "⚠️ Upload a PDF first.", gr.update(), _indexed_md()

    name = Path(file_path).name
    dest = PDF_DIR / name
    if Path(file_path).resolve() != dest.resolve():
        shutil.copy(file_path, dest)  # keep a copy under data/pdfs/

    n = ingest_pdf(dest, with_images=include_figures)
    figs = "with figures" if include_figures else "text only"
    status = f"✅ Ingested **{name}** — {n} chunks ({figs}). Ask away!"
    # refresh the scope dropdown (keep it on the just-added manual) and the indexed list
    return status, gr.update(choices=_source_choices(), value=name), _indexed_md()


# --- UI ---
with gr.Blocks(title="DocLens", theme=gr.themes.Soft(primary_hue="lime")) as demo:
    gr.Markdown(
        "# 📄 DocLens\n"
        "Ask questions about your technical PDF manuals — grounded answers, cited to the "
        "page, with figures the vision model actually read. Fully local."
    )

    with gr.Row():
        scope = gr.Dropdown(
            choices=_source_choices(), value=ALL, label="Manual", scale=2,
        )
        question = gr.Textbox(
            label="Your question",
            placeholder="e.g. How do I connect headphones?  ·  Where is the SD card slot?",
            scale=5,
        )
        submit = gr.Button("Ask", variant="primary", scale=1)

    answer_md = gr.Markdown(label="Answer")
    with gr.Row():
        sources_md = gr.Markdown()
        figures = gr.Gallery(label="Retrieved figures", columns=3, height=280)

    gr.Examples(
        examples=[
            "How do I connect headphones?",
            "Where is the SD card slot on the camera?",
            "How do I set the ISO speed?",
            "What is the capital of India?",  # should refuse — not in the manuals
        ],
        inputs=question,
    )

    with gr.Accordion("➕ Add a new manual", open=False):
        indexed_md = gr.Markdown(_indexed_md())
        with gr.Row():
            upload = gr.File(label="PDF manual", file_types=[".pdf"], type="filepath", scale=3)
            with_figs = gr.Checkbox(label="Include figures (slower — captions each diagram)", value=True, scale=2)
        ingest_btn = gr.Button("Ingest manual", variant="primary")
        ingest_status = gr.Markdown()

    # wiring
    submit.click(ask, inputs=[question, scope], outputs=[answer_md, sources_md, figures])
    question.submit(ask, inputs=[question, scope], outputs=[answer_md, sources_md, figures])
    ingest_btn.click(
        add_manual, inputs=[upload, with_figs], outputs=[ingest_status, scope, indexed_md]
    )


if __name__ == "__main__":
    demo.launch()
