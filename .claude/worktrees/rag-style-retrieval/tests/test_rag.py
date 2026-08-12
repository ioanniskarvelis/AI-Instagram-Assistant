import json

import httpx
import respx

VOYAGE_ENDPOINT = "https://api.voyageai.com/v1/embeddings"


def _write_index(path, entries):
    path.write_text(json.dumps(entries), encoding="utf-8")


def _embed_response(vectors):
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": v, "index": i}
            for i, v in enumerate(vectors)
        ],
        "model": "voyage-3.5",
        "usage": {"total_tokens": 1},
    }


@respx.mock
async def test_retrieve_returns_top_k_by_similarity(monkeypatch, tmp_path):
    from app.config import get_settings
    from app.rag import Example, _load_index, retrieve

    index_path = tmp_path / "rag_index.json"
    _write_index(
        index_path,
        [
            {"question": "close match", "reply": "reply A", "embedding": [1.0, 0.0]},
            {"question": "far match", "reply": "reply B", "embedding": [0.0, 1.0]},
            {"question": "also close", "reply": "reply C", "embedding": [0.9, 0.1]},
        ],
    )
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    monkeypatch.setenv("RAG_INDEX_PATH", str(index_path))
    get_settings.cache_clear()
    _load_index.cache_clear()

    respx.post(VOYAGE_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_embed_response([[1.0, 0.0]]))
    )

    results = await retrieve("query", k=2)

    assert results == [
        Example(question="close match", reply="reply A"),
        Example(question="also close", reply="reply C"),
    ]


@respx.mock
async def test_retrieve_returns_empty_without_api_key(monkeypatch, tmp_path):
    from app.config import get_settings
    from app.rag import _load_index, retrieve

    index_path = tmp_path / "rag_index.json"
    _write_index(index_path, [{"question": "q", "reply": "r", "embedding": [1.0, 0.0]}])
    monkeypatch.setenv("VOYAGE_API_KEY", "")
    monkeypatch.setenv("RAG_INDEX_PATH", str(index_path))
    get_settings.cache_clear()
    _load_index.cache_clear()

    assert await retrieve("query", k=3) == []


@respx.mock
async def test_retrieve_returns_empty_when_index_missing(monkeypatch, tmp_path):
    from app.config import get_settings
    from app.rag import _load_index, retrieve

    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    monkeypatch.setenv("RAG_INDEX_PATH", str(tmp_path / "does-not-exist.json"))
    get_settings.cache_clear()
    _load_index.cache_clear()

    assert await retrieve("query", k=3) == []


@respx.mock
async def test_retrieve_returns_empty_on_voyage_api_error(monkeypatch, tmp_path):
    from app.config import get_settings
    from app.rag import _load_index, retrieve

    index_path = tmp_path / "rag_index.json"
    _write_index(index_path, [{"question": "q", "reply": "r", "embedding": [1.0, 0.0]}])
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    monkeypatch.setenv("RAG_INDEX_PATH", str(index_path))
    get_settings.cache_clear()
    _load_index.cache_clear()

    respx.post(VOYAGE_ENDPOINT).mock(return_value=httpx.Response(500, json={"error": "boom"}))

    assert await retrieve("query", k=3) == []


@respx.mock
async def test_retrieve_returns_empty_on_transport_error(monkeypatch, tmp_path):
    from app.config import get_settings
    from app.rag import _load_index, retrieve

    index_path = tmp_path / "rag_index.json"
    _write_index(index_path, [{"question": "q", "reply": "r", "embedding": [1.0, 0.0]}])
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    monkeypatch.setenv("RAG_INDEX_PATH", str(index_path))
    get_settings.cache_clear()
    _load_index.cache_clear()

    respx.post(VOYAGE_ENDPOINT).mock(side_effect=httpx.ConnectError("boom"))

    assert await retrieve("query", k=3) == []


@respx.mock
async def test_retrieve_returns_empty_for_empty_corpus(monkeypatch, tmp_path):
    from app.config import get_settings
    from app.rag import _load_index, retrieve

    index_path = tmp_path / "rag_index.json"
    _write_index(index_path, [])
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    monkeypatch.setenv("RAG_INDEX_PATH", str(index_path))
    get_settings.cache_clear()
    _load_index.cache_clear()

    assert await retrieve("query", k=3) == []


@respx.mock
async def test_retrieve_returns_empty_on_embedding_dimension_mismatch(monkeypatch, tmp_path):
    from app.config import get_settings
    from app.rag import _load_index, retrieve

    index_path = tmp_path / "rag_index.json"
    _write_index(
        index_path,
        [{"question": "q", "reply": "r", "embedding": [1.0, 0.0]}],
    )
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    monkeypatch.setenv("RAG_INDEX_PATH", str(index_path))
    get_settings.cache_clear()
    _load_index.cache_clear()

    # Index embeddings are 2-D; the query embedding Voyage returns here is
    # 3-D — simulating an index built with a different Voyage model.
    respx.post(VOYAGE_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_embed_response([[1.0, 0.0, 0.0]]))
    )

    assert await retrieve("query", k=3) == []


@respx.mock
async def test_retrieve_returns_empty_for_schema_malformed_index(monkeypatch, tmp_path):
    from app.config import get_settings
    from app.rag import _load_index, retrieve

    index_path = tmp_path / "rag_index.json"
    # Valid JSON, but structurally malformed: one entry is missing "reply",
    # and embeddings have inconsistent lengths across entries.
    index_path.write_text(
        json.dumps(
            [
                {"question": "q1", "embedding": [1.0, 0.0]},
                {"question": "q2", "reply": "r2", "embedding": [0.0, 1.0, 0.5]},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    monkeypatch.setenv("RAG_INDEX_PATH", str(index_path))
    get_settings.cache_clear()
    _load_index.cache_clear()

    assert await retrieve("query", k=3) == []
