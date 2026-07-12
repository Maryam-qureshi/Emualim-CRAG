"""
rag/embedder.py
===============
Shared embedding helper for ingestion and retrieval.

Uses Cohere embed-english-v3.0 (1024 dimensions).
Cohere has explicit input_type parameters for asymmetric retrieval:
  - search_document : used during ingestion
  - search_query    : used during retrieval

This distinction is important — using the wrong input_type
silently degrades retrieval quality.

Install:
    pip install cohere

Add to .env:
    COHERE_API_KEY=your_trial_key_here

IMPORTANT: Atlas Vector Search index must use numDimensions: 1024.
If you already created a 768-dim index, drop it and recreate it.
"""

import os
import time
import logging
from dotenv import load_dotenv
import cohere

load_dotenv()

logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "embed-english-v3.0"
EMBEDDING_DIM   = 1024
MAX_RETRIES     = 3
RETRY_DELAY_S   = 2
MAX_CHARS       = 9000


def _get_client() -> cohere.Client:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "COHERE_API_KEY not found in environment. "
            "Add COHERE_API_KEY=your_key to your .env file."
        )
    return cohere.Client(api_key=api_key)


# ── CORE ──────────────────────────────────────────────────────────────────────

def _embed(text: str, input_type: str) -> list[float]:
    """
    Internal embedding call with retry logic.
    input_type: "search_document" | "search_query"
    """
    text = text.strip()
    if not text:
        raise ValueError("Cannot embed empty text.")

    if len(text) > MAX_CHARS:
        logger.warning(
            f"Text truncated from {len(text)} to {MAX_CHARS} chars before embedding."
        )
        text = text[:MAX_CHARS]

    client = _get_client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embed(
                texts=[text],
                model=EMBEDDING_MODEL,
                input_type=input_type,
                embedding_types=["float"],
            )

            # Cohere returns response.embeddings.float[0]
            embedding = response.embeddings.float[0]

            if len(embedding) != EMBEDDING_DIM:
                raise ValueError(
                    f"Unexpected embedding dimension: {len(embedding)} "
                    f"(expected {EMBEDDING_DIM})"
                )
            return list(embedding)

        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.error(
                    f"Embedding failed after {MAX_RETRIES} attempts: {e}"
                )
                raise
            logger.warning(
                f"Embedding attempt {attempt} failed ({e}). "
                f"Retrying in {RETRY_DELAY_S}s..."
            )
            time.sleep(RETRY_DELAY_S)


def embed_document(text: str) -> list[float]:
    """
    Embed a document chunk for storage during ingestion.
    Uses input_type='search_document'.
    """
    return _embed(text, input_type="search_document")


def embed_query(text: str) -> list[float]:
    """
    Embed a user query at retrieval time.
    Uses input_type='search_query'.
    Must use this — NOT embed_document — for queries.
    """
    return _embed(text, input_type="search_query")


def embed_batch(
    texts: list[str],
    input_type: str = "search_document",
    delay: float = 0.1,
) -> list[list[float]]:
    """
    Embed a list of texts one by one with a small delay between
    calls to stay within Cohere rate limits (trial: 100 calls/min).
    Returns embeddings in the same order as input.
    """
    embeddings = []
    for i, text in enumerate(texts):
        embedding = _embed(text, input_type)
        embeddings.append(embedding)
        if delay > 0 and i < len(texts) - 1:
            time.sleep(delay)
    return embeddings