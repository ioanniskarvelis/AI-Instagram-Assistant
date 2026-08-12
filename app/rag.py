import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

VOYAGE_ENDPOINT = "https://api.voyageai.com/v1/embeddings"
REQUEST_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class Example:
    """One retrieved (question, reply) pair. Style reference only — never a
    source of facts about the current customer or conversation."""

    question: str
    reply: str


@dataclass(frozen=True)
class _Index:
    examples: list[Example]
    embeddings: np.ndarray  # shape (n, dim), one row per example


@lru_cache
def _load_index() -> "_Index | None":
    """Load the built RAG index from disk. Cached like `get_settings`.

    Call `_load_index.cache_clear()` to force a reload — tests do this
    whenever they point RAG_INDEX_PATH at a fixture file.
    """
    path = Path(get_settings().rag_index_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("RAG index not available at %s", path)
        return None
    if not raw:
        return None
    examples = [Example(question=r["question"], reply=r["reply"]) for r in raw]
    embeddings = np.array([r["embedding"] for r in raw], dtype=np.float32)
    return _Index(examples=examples, embeddings=embeddings)


async def _embed_query(text: str, api_key: str, model: str) -> "np.ndarray | None":
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                VOYAGE_ENDPOINT,
                json={"input": [text], "model": model, "input_type": "query"},
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.HTTPError:
        logger.exception("Transport error calling Voyage AI")
        return None

    if response.status_code >= 400:
        logger.error(
            "Voyage AI rejected embed request: %s %s",
            response.status_code,
            response.text,
        )
        return None

    try:
        embedding = response.json()["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError, ValueError):
        logger.exception("Unexpected Voyage AI response shape")
        return None
    return np.array(embedding, dtype=np.float32)


def _top_k(embeddings: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query)
    scores = (embeddings @ query) / (embedding_norms * query_norm)
    ranked = np.argsort(scores)[::-1]
    return [int(i) for i in ranked[:k]]


async def retrieve(text: str, k: int) -> list[Example]:
    """Return the top-k past exchanges most similar to `text`.

    Never raises: returns [] whenever retrieval isn't usable (no API key, no
    index, an embedding call fails) so a retrieval problem never blocks a
    reply going out.
    """
    settings = get_settings()
    if not settings.voyage_api_key:
        return []

    index = _load_index()
    if index is None or not index.examples:
        return []

    query_embedding = await _embed_query(
        text, settings.voyage_api_key, settings.voyage_model
    )
    if query_embedding is None:
        return []

    top_indices = _top_k(index.embeddings, query_embedding, k)
    return [index.examples[i] for i in top_indices]
