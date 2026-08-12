import json

import httpx
import respx

from scripts.rag_build_index import build_index

VOYAGE_ENDPOINT = "https://api.voyageai.com/v1/embeddings"


def _embed_response(vectors):
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": v, "index": i}
            for i, v in enumerate(vectors)
        ],
        "model": "voyage-3.5",
        "usage": {"total_tokens": len(vectors)},
    }


@respx.mock
def test_build_index_embeds_each_entry_as_a_document():
    respx.post(VOYAGE_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_embed_response([[0.1, 0.2], [0.3, 0.4]]))
    )

    entries = [
        {"customer": "q1", "studio_reply_scrubbed": "r1"},
        {"customer": "q2", "studio_reply_scrubbed": "r2"},
    ]
    indexed = build_index(entries, api_key="test-key")

    assert indexed == [
        {"question": "q1", "reply": "r1", "embedding": [0.1, 0.2]},
        {"question": "q2", "reply": "r2", "embedding": [0.3, 0.4]},
    ]
    request_body = json.loads(respx.calls.last.request.content)
    assert request_body["input"] == ["q1", "q2"]
    assert request_body["input_type"] == "document"
    assert respx.calls.last.request.headers["Authorization"] == "Bearer test-key"


@respx.mock
def test_build_index_pairs_entries_by_response_index_not_position():
    # Voyage's response `data` array is returned out of order (index 1
    # before index 0). build_index must pair each input text with the
    # embedding whose own `index` field matches, not with whatever
    # embedding happens to occupy that position in the response.
    out_of_order_response = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
            {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
        ],
        "model": "voyage-3.5",
        "usage": {"total_tokens": 2},
    }
    respx.post(VOYAGE_ENDPOINT).mock(return_value=httpx.Response(200, json=out_of_order_response))

    entries = [
        {"customer": "q1", "studio_reply_scrubbed": "r1"},
        {"customer": "q2", "studio_reply_scrubbed": "r2"},
    ]
    indexed = build_index(entries, api_key="test-key")

    assert indexed == [
        {"question": "q1", "reply": "r1", "embedding": [0.1, 0.2]},
        {"question": "q2", "reply": "r2", "embedding": [0.3, 0.4]},
    ]


@respx.mock
def test_build_index_batches_large_corpora():
    from scripts import rag_build_index

    def _respond(request):
        batch_input = json.loads(request.content)["input"]
        return httpx.Response(200, json=_embed_response([[0.0, 0.0]] * len(batch_input)))

    respx.post(VOYAGE_ENDPOINT).mock(side_effect=_respond)

    entries = [
        {"customer": f"q{i}", "studio_reply_scrubbed": f"r{i}"}
        for i in range(rag_build_index.BATCH_SIZE + 1)
    ]
    indexed = build_index(entries, api_key="test-key")

    assert len(indexed) == rag_build_index.BATCH_SIZE + 1
    assert respx.calls.call_count == 2
