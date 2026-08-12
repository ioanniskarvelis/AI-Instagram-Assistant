import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

EMBEDDING_ENDPOINT = "https://openrouter.ai/api/v1/embeddings"
REQUEST_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class Example:
    """One retrieved (question, reply) pair. Style reference only — never a
    source of facts about the current customer or conversation."""

    question: str
    reply: str


@dataclass(frozen=True)
class ScoredExample:
    """An `Example` plus the cosine similarity that ranked it — the
    observability-only counterpart of `retrieve()`'s plain result, so a trace
    can show *why* an example was picked, not just that it was."""

    example: Example
    score: float


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
        if not raw:
            return None
        examples = [Example(question=r["question"], reply=r["reply"]) for r in raw]
        embeddings = np.array([r["embedding"] for r in raw], dtype=np.float32)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.warning("RAG index not available or malformed at %s", path)
        return None
    return _Index(examples=examples, embeddings=embeddings)


async def _embed_query(text: str, api_key: str, model: str) -> "np.ndarray | None":
    """Embed one query string via OpenRouter's embeddings endpoint.

    Goes through OpenRouter rather than a direct Voyage AI account, so the
    request uses OpenRouter's generic schema (model, input) rather than
    Voyage-native extras — in particular, there's no confirmed way to pass
    Voyage's asymmetric input_type=query/document distinction through
    OpenRouter, so both this call and the index-build script embed plain
    text with no input_type set.
    """
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                EMBEDDING_ENDPOINT,
                json={"input": [text], "model": model},
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.HTTPError:
        logger.exception("Transport error calling the embeddings endpoint")
        return None

    if response.status_code >= 400:
        logger.error(
            "Embeddings endpoint rejected embed request: %s %s",
            response.status_code,
            response.text,
        )
        return None

    try:
        embedding = response.json()["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError, ValueError):
        logger.exception("Unexpected embeddings response shape")
        return None
    return np.array(embedding, dtype=np.float32)


def _top_k(embeddings: np.ndarray, query: np.ndarray, k: int) -> list[tuple[int, float]]:
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = (embeddings @ query) / (embedding_norms * query_norm)
    scores = np.nan_to_num(scores, nan=-np.inf)
    ranked = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in ranked]


async def retrieve_scored(text: str, k: int) -> list[ScoredExample]:
    """Return the top-k past exchanges most similar to `text`, with scores.

    Never raises: returns [] whenever retrieval isn't usable (no API key, no
    index, an embedding call fails) so a retrieval problem never blocks a
    reply going out. `retrieve()` is a thin wrapper over this for callers
    that only need the examples, not the scores.
    """
    settings = get_settings()
    if not settings.openrouter_api_key:
        return []

    index = _load_index()
    if index is None or not index.examples:
        return []

    query_embedding = await _embed_query(
        text, settings.openrouter_api_key, settings.embedding_model
    )
    if query_embedding is None:
        return []

    try:
        ranked = _top_k(index.embeddings, query_embedding, k)
    except Exception:
        logger.exception("Ranking failed against the loaded index")
        return []

    return [ScoredExample(index.examples[i], score) for i, score in ranked]


async def retrieve(text: str, k: int) -> list[Example]:
    """Return the top-k past exchanges most similar to `text`."""
    return [scored.example for scored in await retrieve_scored(text, k)]
