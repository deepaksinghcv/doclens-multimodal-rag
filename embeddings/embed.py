"""
Box 3 — EMBED.

Turn text into 384-dim vectors using bge-small-en-v1.5.

Two functions on purpose (Decision D4): bge expects an instruction prefix on the QUERY
only, never on the documents. Mixing these up silently hurts recall.

The model is heavy to load, so we load it ONCE (lazy singleton) and reuse it for every call.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBED_MODEL, QUERY_PREFIX

# Module-level cache for the loaded model. None until first use.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it (lazy singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed CHUNK texts (no prefix).

    Args:
        texts: the raw chunk texts.

    Returns:
        A list of 384-dim vectors (one per input), L2-normalized.
    """
    model = get_model()
    doc_embeds = model.encode(
        sentences=texts,            #we dont have to loop, batching happens internally
        batch_size=32,              #batch size
        normalize_embeddings=True,  #unit vectors, for easing similarity calculations
        convert_to_numpy=True       #chromadb expects numpy array, not tensors
    )  
    return np.asarray(doc_embeds).tolist() #chromadb expects numpy array, not tensors


def embed_query(query: str) -> list[float]:
    """Embed a QUERY (WITH the bge instruction prefix, per D4).

    Args:
        query: the user's question.

    Returns:
        A single 384-dim vector, L2-normalized.
    """
    model = get_model()
    q_embed = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return np.array(q_embed).tolist()
    


if __name__ == "__main__":
    # Smoke test: embed a couple of chunks and a query; check shapes and that a
    # semantically-close pair scores higher than a far pair.
    import numpy as np

    docs = embed_documents(["Wash the filter under cold running water.",
                            "The keyboard supports 200 preset voices."])
    q = embed_query("How do I clean the filter?")

    print(f"doc vector dim: {len(docs[0])}  (expect 384)")
    print(f"query vector dim: {len(q)}  (expect 384)")

    q = np.array(q)
    sim_close = q @ np.array(docs[0])   # filter cleaning  (should be higher)
    sim_far = q @ np.array(docs[1])     # preset voices    (should be lower)
    print(f"similarity to 'wash the filter': {sim_close:.3f}")
    print(f"similarity to 'preset voices' : {sim_far:.3f}")
    print("PASS" if sim_close > sim_far else "FAIL — something's off")
