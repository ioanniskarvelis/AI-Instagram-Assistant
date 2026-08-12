"""Embed the approved RAG corpus and write the runtime index.

Usage, after reviewing data/rag_corpus_review.jsonl and saving your edits to
data/rag_corpus_approved.jsonl:

    OPENROUTER_API_KEY=... python -m scripts.rag_build_index

Writes data/rag_index.json, which app/rag.py loads at runtime. Embeddings go
through OpenRouter (openrouter.ai/api/v1/embeddings) rather than a direct
Voyage AI account — OpenRouter's documented request schema is just
model/input/encoding_format, with no confirmed pass-through for Voyage's
native input_type=query/document distinction, so this script and app/rag.py
both embed plain text with no input_type set. See
docs/superpowers/specs/2026-08-12-rag-style-retrieval-design.md.
"""
import json
import os
from pathlib import Path

import httpx

EMBEDDING_ENDPOINT = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "voyageai/voyage-4")
BATCH_SIZE = 128
REQUEST_TIMEOUT_SECONDS = 30.0

APPROVED_PATH = Path("data/rag_corpus_approved.jsonl")
INDEX_PATH = Path("data/rag_index.json")


def _read_approved(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _embed_batch(texts: list[str], api_key: str) -> list[list[float]]:
    response = httpx.post(
        EMBEDDING_ENDPOINT,
        json={"input": texts, "model": EMBEDDING_MODEL},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = sorted(response.json()["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in data]


def build_index(entries: list[dict], api_key: str) -> list[dict]:
    indexed: list[dict] = []
    for start in range(0, len(entries), BATCH_SIZE):
        batch = entries[start : start + BATCH_SIZE]
        embeddings = _embed_batch([entry["customer"] for entry in batch], api_key)
        for entry, embedding in zip(batch, embeddings, strict=True):
            indexed.append(
                {
                    "question": entry["customer"],
                    "reply": entry["studio_reply_scrubbed"],
                    "embedding": embedding,
                }
            )
    return indexed


def main() -> None:
    api_key = os.environ["OPENROUTER_API_KEY"]
    entries = _read_approved(APPROVED_PATH)
    indexed = build_index(entries, api_key)
    INDEX_PATH.write_text(json.dumps(indexed, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(indexed)} embedded examples to {INDEX_PATH}")


if __name__ == "__main__":
    main()
