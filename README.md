# DocLens: Local Multimodal RAG for Technical Documents

> Ask questions about manuals, diagrams, and technical PDFs using a fully local Retrieval-Augmented Generation (RAG) pipeline.

---

## Overview

DocLens is a local multimodal RAG assistant designed to understand technical documents such as user manuals, product guides, and engineering documentation.

Given a PDF and a natural language question, DocLens retrieves relevant sections from the document and generates grounded answers with citations.

Example:

### Input

```text
PDF: Dyson_Vacuum_Manual.pdf

Question:
How do I clean the filter?
```

### Output

```text
To clean the filter:

1. Remove the filter assembly.
2. Wash the filter under cold running water.
3. Allow it to dry completely for 24 hours before reinstalling.

Source:
Page 12
```

Unlike generic chatbots, DocLens grounds its responses using retrieved evidence from the uploaded document.

---

## Motivation

Modern AI assistants often hallucinate when answering domain-specific questions.

Retrieval-Augmented Generation addresses this problem by allowing language models to access external knowledge at inference time.

Technical documents present additional challenges:

- Long context lengths
- Complex formatting
- Tables and figures
- Procedural instructions
- Cross-page references

DocLens explores how multimodal RAG systems can improve reliability while remaining lightweight enough to run on consumer hardware.

---

## Features

### Current Features

- PDF ingestion
- Text extraction
- Intelligent document chunking
- Dense retrieval using embeddings
- Local vector database
- Grounded answer generation
- Source page citations
- Fully local execution

### Planned Features

- Figure and diagram understanding
- Image retrieval
- Table question answering
- Hybrid retrieval (BM25 + dense search)
- Citation highlighting
- Interactive Gradio interface
- Comparative evaluation of RAG vs RAG + LoRA

---

## System Architecture

```text
                     ┌─────────────┐
                     │ PDF Manual  │
                     └──────┬──────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │ Text Extraction │
                  └────────┬────────┘
                           │
                           ▼
                   ┌─────────────┐
                   │ Chunking    │
                   └─────┬───────┘
                         │
                         ▼
              ┌────────────────────┐
              │ Embedding Model    │
              │ (BGE Small)        │
              └─────────┬──────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Chroma VectorDB │
               └────────┬────────┘
                        │
          User Question │
                        ▼
               ┌─────────────────┐
               │ Retrieval       │
               │ Top-K Chunks    │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Qwen Generator  │
               └────────┬────────┘
                        │
                        ▼
             Grounded Answer + Citation
```

---

## Tech Stack

### Document Processing

- PyMuPDF

### Embeddings

- BAAI/bge-small-en-v1.5

### Vector Database

- ChromaDB

### Generation

- Qwen 3 (via Ollama)

### UI (Planned)

- Gradio

### Language

- Python 3.11+

---

## Project Structure

```text
doclens/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── pdfs/
│   ├── extracted_images/
│   └── evaluation/
│
├── ingestion/
│   ├── parse_pdf.py
│   ├── extract_images.py
│   └── chunker.py
│
├── embeddings/
│   └── embed.py
│
├── vectordb/
│   └── chroma_store.py
│
├── retrieval/
│   └── retrieve.py
│
├── generation/
│   └── answer.py
│
├── evaluation/
│   ├── retrieval_eval.py
│   └── generation_eval.py
│
├── app/
│   ├── cli.py
│   └── gradio_app.py
│
└── notebooks/
```

---

## Implementation Roadmap

### Phase 1: Text RAG MVP

Goal:

Build a working retrieval system for technical PDFs.

Tasks:

- [ ] Extract text using PyMuPDF
- [ ] Chunk documents
- [ ] Generate embeddings
- [ ] Store embeddings in ChromaDB
- [ ] Retrieve relevant chunks

Deliverable:

Question → Retrieved Evidence

---

### Phase 2: Grounded Question Answering

Goal:

Generate answers from retrieved context.

Tasks:

- [ ] Integrate Qwen through Ollama
- [ ] Construct prompts
- [ ] Return cited answers
- [ ] Handle insufficient context gracefully

Deliverable:

Question → Answer + Page Citation

---

### Phase 3: Multimodal Extension

Goal:

Incorporate visual information.

Tasks:

- [ ] Extract images and diagrams
- [ ] Generate image embeddings
- [ ] Retrieve text and visual context
- [ ] Use Qwen-VL for reasoning

Deliverable:

Question → Answer grounded in text and figures

---

### Phase 4: Evaluation

Goal:

Measure system quality.

Tasks:

- [ ] Build evaluation dataset
- [ ] Compute Retrieval Recall@K
- [ ] Evaluate answer correctness
- [ ] Measure citation faithfulness

Deliverable:

Quantitative evaluation report

---

## Evaluation Strategy

### Retrieval Metrics

- Recall@K
- Mean Reciprocal Rank (MRR)

### Generation Metrics

- Exact Match
- Answer Accuracy
- Faithfulness
- Citation Correctness

### Example Evaluation Sample

```json
{
    "question": "How do I replace the air filter?",
    "expected_answer": "Remove the old filter and insert the replacement.",
    "expected_page": 18
}
```

---

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Ollama:

```bash
ollama serve
```

Download Qwen:

```bash
ollama pull qwen3:4b
```

Run ingestion:

```bash
python ingestion/parse_pdf.py
```

Build the vector database:

```bash
python vectordb/chroma_store.py
```

Launch the assistant:

```bash
python app/cli.py
```

---

## Future Directions

Potential extensions include:

- Research paper assistants
- Enterprise document search
- Compliance assistants
- Contract understanding
- Table reasoning
- Diagram question answering
- Agentic workflows over documentation
- Fine-tuning using RAG failure cases

---

## Key Learnings

This project explores practical aspects of modern AI systems:

- Retrieval-Augmented Generation
- Embedding-based search
- Vector databases
- Prompt engineering
- Grounded generation
- Citation-aware responses
- Multimodal reasoning
- Local-first deployment

---

## Why DocLens?

DocLens was built to answer a simple question:

> Can we build a reliable document assistant that runs locally, understands technical knowledge, and cites its sources?

This project demonstrates that effective AI systems are often built not through larger models alone, but through thoughtful system design combining retrieval, reasoning, and grounding.