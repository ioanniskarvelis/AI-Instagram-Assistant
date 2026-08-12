# RAG Style Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the assistant access to the studio's real past DM replies as a
style/tone reference at generation time, without ever letting stale prices,
booking times, or unverified facts reach the model.

**Architecture:** Two offline scripts (run on the host, never in the
container) turn the raw `inbox/` export into a small embedded index after a
human review checkpoint. One new runtime module (`app/rag.py`) loads that
index and does in-memory cosine-similarity retrieval per incoming message,
feeding the top-k results into `app/llm.py` as a labeled system-prompt block.

**Tech Stack:** FastAPI, `httpx` (Voyage AI embedding calls, same pattern as
the existing Graph API / Anthropic calls), `numpy` (new — flat in-memory
cosine similarity, no vector database), Python stdlib `json`/`re` for the
offline extraction script.

**Spec:** `docs/superpowers/specs/2026-08-12-rag-style-retrieval-design.md`

## Global Constraints

- Retrieval is style-only. It must never be used to answer a factual question
  from historical data, and it must never weaken the existing hard rules in
  `SYSTEM_PROMPT` (no prices, no bookings, no unverified facts).
- Every new runtime failure mode (missing index, empty corpus, Voyage AI
  down, no API key) degrades to `retrieve()` returning `[]` — never raises,
  never blocks a reply from going out. This matches the existing
  never-raise contract of `app/instagram.py` and `app/llm.py`.
- No vector database. Corpus size is at most a few thousand pairs; a flat
  `numpy` matrix with cosine similarity is the whole retrieval engine.
- `inbox/`, `data/rag_corpus_review.jsonl`, `data/rag_corpus_approved.jsonl`,
  and `data/rag_index.json` never reach git or the Docker build context.
- The two offline scripts (`scripts/rag_extract.py`,
  `scripts/rag_build_index.py`) run on the host with the existing `.venv`.
  Only the built `rag_index.json` artifact reaches the running container,
  via a read-only bind mount — no image rebuild needed to refresh the corpus.
- Retrieved examples are appended to the **system** prompt only, after the
  existing hard rules, never spliced into the `messages` array.
- The Docker image must stay well under the existing 300MB budget (currently
  ~53MB); `numpy` is the only new runtime dependency.

---

### Task 1: RAG configuration settings

**Files:**
- Modify: `app/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.voyage_api_key: str` (default `""`),
  `Settings.voyage_model: str` (default `"voyage-3.5"`),
  `Settings.rag_top_k: int` (default `3`),
  `Settings.rag_index_path: str` (default `"./data/rag_index.json"`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_rag_settings_apply_documented_defaults():
    from app.config import get_settings

    settings = get_settings()
    assert settings.voyage_api_key == ""
    assert settings.voyage_model == "voyage-3.5"
    assert settings.rag_top_k == 3
    assert settings.rag_index_path.endswith("rag_index.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_config.py::test_rag_settings_apply_documented_defaults -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'voyage_api_key'`

- [ ] **Step 3: Add the settings**

In `app/config.py`, add after the `assistant_admin_key` field (inside the
`Settings` class):

```python
    # Style retrieval from past DM history (RAG). Leave VOYAGE_API_KEY blank
    # to keep the feature off — the assistant behaves exactly as without it.
    voyage_api_key: str = ""
    voyage_model: str = "voyage-3.5"
    rag_top_k: int = 3
    rag_index_path: str = "./data/rag_index.json"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: PASS (all tests in the file, including the new one)

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat(config): add RAG settings (Voyage API, top-k, index path)"
```

---

### Task 2: `app/rag.py` — index loading and retrieval

**Files:**
- Create: `app/rag.py`
- Test: `tests/test_rag.py`
- Modify: `tests/conftest.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `Settings.voyage_api_key`, `Settings.voyage_model`,
  `Settings.rag_index_path` from Task 1.
- Produces: `Example` (frozen dataclass: `question: str`, `reply: str`);
  `async retrieve(text: str, k: int) -> list[Example]` — never raises,
  returns `[]` on any failure; `_load_index()` (`lru_cache`-decorated,
  `_load_index.cache_clear()` resets it, same pattern as
  `app.config.get_settings`).

- [ ] **Step 1: Add numpy to requirements**

In `requirements.txt`, add a new line:

```
numpy==2.2.1
```

- [ ] **Step 2: Install it locally**

Run: `.venv/Scripts/python -m pip install -r requirements.txt -r requirements-dev.txt`
Expected: numpy installs alongside the existing dependencies.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_rag.py`:

```python
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
```

This will fail to import (`app.rag` does not exist yet).

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_rag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.rag'`

- [ ] **Step 5: Write `app/rag.py`**

```python
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
```

- [ ] **Step 6: Wire cache-clearing into the shared test fixture**

`_load_index` is process-cached like `get_settings`, so any test that wants a
specific index (or a missing one) must clear it, and every other test needs
it to start clean. Modify `tests/conftest.py`'s `env` fixture:

```python
@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("IG_VERIFY_TOKEN", VERIFY_TOKEN)
    monkeypatch.setenv("IG_USER_ACCESS_TOKEN", ACCESS_TOKEN)
    monkeypatch.setenv("IG_APP_SECRET", APP_SECRET)
    monkeypatch.setenv("IG_ACCOUNT_ID", ACCOUNT_ID)
    # Pin settings that have a documented default so tests aren't affected by
    # whatever the developer's real .env happens to contain.
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("ALLOWED_SENDER_IDS", "")
    monkeypatch.setenv("CANNED_REPLY", CANNED_REPLY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "history.db"))
    monkeypatch.setenv("ASSISTANT_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setenv("VOYAGE_API_KEY", "")
    monkeypatch.setenv("VOYAGE_MODEL", "voyage-3.5")
    monkeypatch.setenv("RAG_TOP_K", "3")
    monkeypatch.setenv("RAG_INDEX_PATH", str(tmp_path / "rag_index.json"))

    from app.config import get_settings
    from app.rag import _load_index

    get_settings.cache_clear()
    _load_index.cache_clear()
    yield
    get_settings.cache_clear()
    _load_index.cache_clear()
```

(Only the four new `monkeypatch.setenv` calls, the `from app.rag import
_load_index` line, and the two `_load_index.cache_clear()` calls are new —
everything else in the fixture is unchanged.)

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_rag.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 8: Run the full test suite to check for regressions**

Run: `.venv/Scripts/python -m pytest -v`
Expected: PASS (all existing tests still pass — `RAG_INDEX_PATH` defaults to
a nonexistent tmp file and `VOYAGE_API_KEY` defaults to empty, so `retrieve`
is inert unless a test opts in)

- [ ] **Step 9: Commit**

```bash
git add app/rag.py tests/test_rag.py tests/conftest.py requirements.txt
git commit -m "feat(rag): add index loading and top-k retrieval via Voyage AI"
```

---

### Task 3: Inject retrieved examples into the system prompt

**Files:**
- Modify: `app/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `Example` from `app.rag` (Task 2).
- Produces: `generate_reply(turns: list[Turn], examples: list[Example] | None = None) -> str | None`
  — the `examples` parameter is new and optional; existing callers that pass
  only `turns` are unaffected.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_llm.py`:

```python
@respx.mock
async def test_generate_reply_includes_style_examples_after_rules():
    from app.history import Turn
    from app.llm import SYSTEM_PROMPT, generate_reply
    from app.rag import Example

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    examples = [
        Example(question="Πόσο κοστίζει;", reply="Στείλε φωτο και σου λέμε [price]"),
    ]
    await generate_reply([Turn(role="user", text="γεια")], examples)

    body = json.loads(route.calls.last.request.content)
    system_text = body["system"][0]["text"]
    assert system_text.startswith(SYSTEM_PROMPT)
    style_index = system_text.index("STYLE REFERENCE")
    assert style_index > len(SYSTEM_PROMPT)
    assert "Πόσο κοστίζει;" in system_text
    assert "Στείλε φωτο και σου λέμε [price]" in system_text


@respx.mock
async def test_generate_reply_without_examples_matches_todays_prompt():
    from app.history import Turn
    from app.llm import SYSTEM_PROMPT, generate_reply

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    await generate_reply([Turn(role="user", text="γεια")])

    body = json.loads(route.calls.last.request.content)
    assert body["system"][0]["text"] == SYSTEM_PROMPT
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_llm.py -v`
Expected: FAIL — `test_generate_reply_includes_style_examples_after_rules`
fails with a `TypeError` (`generate_reply` doesn't accept a second argument
yet); the no-examples test already passes since it matches current behavior.

- [ ] **Step 3: Modify `app/llm.py`**

Add the import and the rendering helper near the top (after the existing
imports), and update `generate_reply`'s signature and system-block
construction:

```python
from app.config import get_settings
from app.history import Turn
from app.rag import Example

# ... SYSTEM_PROMPT unchanged ...

STYLE_BLOCK_HEADER = "--- STYLE REFERENCE (not this conversation, tone/phrasing only) ---"
STYLE_BLOCK_FOOTER = (
    "Ignore any prices, dates, or specifics in these examples — the rules "
    "above still apply."
)


def _render_style_block(examples: list[Example]) -> str:
    parts = [STYLE_BLOCK_HEADER]
    for i, example in enumerate(examples, start=1):
        parts.append(
            f"Example {i}\nCustomer: {example.question}\nStudio: {example.reply}"
        )
    parts.append(STYLE_BLOCK_FOOTER)
    return "\n\n".join(parts)


async def generate_reply(
    turns: list[Turn], examples: list[Example] | None = None
) -> str | None:
    """Generate a reply from the conversation window.

    Returns None on any failure. Never raises: the webhook must return 200 to
    Meta whether or not generation worked, and the caller falls back to the
    canned reply.
    """
    system_text = SYSTEM_PROMPT
    if examples:
        system_text = f"{SYSTEM_PROMPT}\n\n{_render_style_block(examples)}"

    try:
        settings = get_settings()
        response = await _client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            output_config={"effort": settings.llm_effort},
            messages=[{"role": turn.role, "content": turn.text} for turn in turns],
        )
    except anthropic.APIError:
        logger.exception("Anthropic call failed")
        return None
    except Exception:
        logger.exception("Unexpected error generating reply")
        return None

    # ... rest of the function (stop_reason checks, text extraction) is unchanged ...
```

Only the signature, the new module-level constants/helper, and the
`system_text` construction change — the failure handling and text-extraction
logic below the `try` block stay exactly as they are today.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_llm.py -v`
Expected: PASS (all tests, including the two new ones)

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `.venv/Scripts/python -m pytest -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/llm.py tests/test_llm.py
git commit -m "feat(llm): append retrieved style examples to the system prompt"
```

---

### Task 4: Wire retrieval into the webhook flow

**Files:**
- Modify: `app/webhook.py`
- Test: `tests/test_webhook_receive.py`

**Interfaces:**
- Consumes: `retrieve(text: str, k: int) -> list[Example]` from `app.rag`
  (Task 2); `generate_reply(turns, examples)` from `app.llm` (Task 3);
  `Settings.rag_top_k` from `app.config` (Task 1).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_webhook_receive.py` (near the top, alongside the other
imports, add `from app.config import get_settings` inside the test body as
the existing allowlist tests already do):

```python
@respx.mock
def test_style_examples_reach_the_system_prompt(client, monkeypatch, tmp_path):
    from app.config import get_settings
    from app.rag import _load_index

    index_path = tmp_path / "rag_index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "question": "Γεια σας",
                    "reply": "Καλησπέρα, πώς μπορούμε να βοηθήσουμε;",
                    "embedding": [1.0, 0.0],
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    monkeypatch.setenv("RAG_INDEX_PATH", str(index_path))
    get_settings.cache_clear()
    _load_index.cache_clear()

    respx.post("https://api.voyageai.com/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": [1.0, 0.0], "index": 0}],
                "model": "voyage-3.5",
                "usage": {"total_tokens": 1},
            },
        )
    )
    llm = _mock_llm()
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    _post(client, _body({"mid": "m1", "text": "Γεια σας"}))

    body = json.loads(llm.calls.last.request.content)
    assert "Καλησπέρα, πώς μπορούμε να βοηθήσουμε;" in body["system"][0]["text"]


@respx.mock
def test_no_style_examples_when_voyage_key_unset(client):
    """Today's behavior, unchanged: no VOYAGE_API_KEY means no RAG block."""
    llm = _mock_llm()
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    _post(client, _body({"mid": "m1", "text": "Γεια σας"}))

    body = json.loads(llm.calls.last.request.content)
    assert "STYLE REFERENCE" not in body["system"][0]["text"]
```

- [ ] **Step 2: Run tests to verify the first one fails**

Run: `.venv/Scripts/python -m pytest tests/test_webhook_receive.py -v`
Expected: `test_style_examples_reach_the_system_prompt` FAILS (retrieval is
never called yet, so the style block never appears);
`test_no_style_examples_when_voyage_key_unset` already PASSES (matches
today's behavior).

- [ ] **Step 3: Modify `app/webhook.py`**

Add the import:

```python
from app.rag import retrieve
```

Change the block that builds the reply (between reading the window and
calling `generate_reply`):

```python
        window = await recent(sender_id, settings.history_window_messages)
        if not window:
            # Storage is unavailable; answer the message in front of us rather
            # than going silent.
            window = [Turn(role="user", text=text)]

        examples = await retrieve(text, settings.rag_top_k)
        reply = await generate_reply(window, examples)
```

(Only the new `examples = await retrieve(...)` line and passing `examples`
into `generate_reply` are new — nothing else in the handler changes.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_webhook_receive.py -v`
Expected: PASS (both new tests)

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `.venv/Scripts/python -m pytest -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/webhook.py tests/test_webhook_receive.py
git commit -m "feat(webhook): retrieve style examples before generating a reply"
```

---

### Task 5: Offline extraction script (`scripts/rag_extract.py`)

**Files:**
- Create: `scripts/__init__.py` (empty)
- Create: `scripts/rag_extract.py`
- Test: `tests/test_rag_extract.py`

**Interfaces:**
- Produces (used only by a human running the script, and by Task 6's
  reviewer workflow — no runtime code imports this module):
  `Pair` (frozen dataclass: `thread_id: str`, `customer: str`,
  `studio_reply_scrubbed: str`); `fix_mojibake(text: str) -> str`;
  `scrub_pricing(text: str) -> str`; `extract_pairs(thread_id: str,
  messages: list[dict]) -> list[Pair]`; `dedupe_pairs(pairs: list[Pair]) ->
  list[Pair]`; `main() -> None`.

- [ ] **Step 1: Create the package marker**

Create `scripts/__init__.py` (empty file).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_rag_extract.py`:

```python
import json

from scripts.rag_extract import (
    Pair,
    _collapse_turns,
    _load_thread_messages,
    dedupe_pairs,
    extract_pairs,
    fix_mojibake,
    scrub_pricing,
)


def _mojibake(text: str) -> str:
    """Simulate Meta's export bug: UTF-8 bytes reinterpreted as latin1."""
    return text.encode("utf-8").decode("latin1")


def test_fix_mojibake_restores_original_text():
    original = "Καλησπέρα σας! Πόσο κοστίζει;"
    assert fix_mojibake(_mojibake(original)) == original


def test_fix_mojibake_leaves_plain_ascii_unchanged():
    assert fix_mojibake("hello") == "hello"


def test_scrub_pricing_replaces_euro_symbol_amount():
    assert scrub_pricing("Θα σου κοστίσει 150€ περίπου") == "Θα σου κοστίσει [price] περίπου"


def test_scrub_pricing_replaces_euro_word_amount():
    assert scrub_pricing("κοστίζει 150 ευρώ") == "κοστίζει [price]"


def test_scrub_pricing_replaces_range():
    assert scrub_pricing("κοστίζει 100-150 ευρώ") == "κοστίζει [price]"


def test_scrub_pricing_replaces_booking_time():
    assert scrub_pricing("Ελα αύριο στις 5 μ.μ.") == "Ελα αύριο στις [price]"


def test_scrub_pricing_replaces_english_euro_amount():
    assert (
        scrub_pricing("It will cost around 150 euros for that size")
        == "It will cost around [price] for that size"
    )


def test_scrub_pricing_leaves_unrelated_numbers_alone():
    assert scrub_pricing("θέλω tattoo 10 πόντους") == "θέλω tattoo 10 πόντους"


def test_collapse_turns_merges_consecutive_same_sender():
    messages = [
        {"sender_name": "Maria", "content": "Γεια", "timestamp_ms": 1},
        {"sender_name": "Maria", "content": "θέλω ραντεβού", "timestamp_ms": 2},
        {"sender_name": "2310tattoo studio by Christina", "content": "Καλησπέρα!", "timestamp_ms": 3},
    ]
    assert _collapse_turns(messages) == [
        ("customer", "Γεια\nθέλω ραντεβού"),
        ("studio", "Καλησπέρα!"),
    ]


def test_collapse_turns_skips_messages_without_content():
    messages = [
        {"sender_name": "Maria", "content": None, "timestamp_ms": 1},
        {"sender_name": "Maria", "content": "Γεια", "timestamp_ms": 2},
    ]
    assert _collapse_turns(messages) == [("customer", "Γεια")]


def test_extract_pairs_pairs_customer_then_studio():
    messages = [
        {"sender_name": "Maria", "content": "Πόσο κοστίζει ένα μικρό τατουάζ;", "timestamp_ms": 1},
        {
            "sender_name": "2310tattoo studio by Christina",
            "content": "Στείλε φωτο για να δούμε, κοστίζει 100€",
            "timestamp_ms": 2,
        },
    ]
    assert extract_pairs("THREAD1", messages) == [
        Pair(
            thread_id="THREAD1",
            customer="Πόσο κοστίζει ένα μικρό τατουάζ;",
            studio_reply_scrubbed="Στείλε φωτο για να δούμε, κοστίζει [price]",
        )
    ]


def test_extract_pairs_skips_studio_only_thread():
    messages = [
        {"sender_name": "2310tattoo studio by Christina", "content": "Καλησπέρα!", "timestamp_ms": 1},
    ]
    assert extract_pairs("THREAD1", messages) == []


def test_extract_pairs_drops_short_turns():
    messages = [
        {"sender_name": "Maria", "content": "ok", "timestamp_ms": 1},
        {"sender_name": "2310tattoo studio by Christina", "content": "😊", "timestamp_ms": 2},
    ]
    assert extract_pairs("THREAD1", messages) == []


def test_dedupe_pairs_keeps_first_occurrence_only():
    pairs = [
        Pair(thread_id="T1", customer="q1", studio_reply_scrubbed="same reply"),
        Pair(thread_id="T2", customer="q2", studio_reply_scrubbed="same reply"),
        Pair(thread_id="T3", customer="q3", studio_reply_scrubbed="different"),
    ]
    assert dedupe_pairs(pairs) == [
        Pair(thread_id="T1", customer="q1", studio_reply_scrubbed="same reply"),
        Pair(thread_id="T3", customer="q3", studio_reply_scrubbed="different"),
    ]


def test_load_thread_messages_merges_and_sorts_multiple_files(tmp_path):
    thread_dir = tmp_path / "THREAD1"
    thread_dir.mkdir()
    (thread_dir / "message_2.json").write_text(
        json.dumps({"messages": [{"sender_name": "Maria", "content": "second", "timestamp_ms": 2}]}),
        encoding="utf-8",
    )
    (thread_dir / "message_1.json").write_text(
        json.dumps({"messages": [{"sender_name": "Maria", "content": "first", "timestamp_ms": 1}]}),
        encoding="utf-8",
    )
    messages = _load_thread_messages(thread_dir)
    assert [m["content"] for m in messages] == ["first", "second"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_rag_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.rag_extract'`

- [ ] **Step 4: Write `scripts/rag_extract.py`**

```python
"""Extract (customer, studio reply) pairs from the raw Meta DM export.

Usage:
    python -m scripts.rag_extract

Reads every thread in ./inbox/ and writes candidate pairs to
data/rag_corpus_review.jsonl for human review. Nothing here calls an
external API or touches the running application — this is a one-time (or
occasional) offline step. See
docs/superpowers/specs/2026-08-12-rag-style-retrieval-design.md.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

STUDIO_NAME = "2310tattoo studio by Christina"
MIN_TURN_LENGTH = 3

INBOX_DIR = Path("inbox")
OUTPUT_PATH = Path("data/rag_corpus_review.jsonl")

_CURRENCY_PATTERN = re.compile(
    r"""
    (?:\d+(?:[.,]\d+)?\s*[-–]\s*)?   # optional range start, e.g. "100-"
    \d+(?:[.,]\d+)?                  # the number
    \s*
    (?:€|ευρ[ωώ]|euro s?)            # currency symbol or word
    """,
    re.IGNORECASE | re.VERBOSE,
)
_EURO_PREFIX_PATTERN = re.compile(r"€\s*\d+(?:[.,]\d+)?", re.IGNORECASE)
_TIME_PATTERN = re.compile(
    r"\d{1,2}(?::\d{2})?\s*(?:π\.μ\.|μ\.μ\.|το πρωί|το απόγευμα|το βράδυ|am|pm)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Pair:
    thread_id: str
    customer: str
    studio_reply_scrubbed: str


def fix_mojibake(text: str) -> str:
    """Undo Meta's export bug: UTF-8 bytes were written out as if latin1."""
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def scrub_pricing(text: str) -> str:
    """Replace price and booking-time mentions with a neutral placeholder.

    Heuristic, not exhaustive — the human review step (data/rag_corpus_review.jsonl)
    is the real safety net; the hard rule against quoting prices lives in
    app/llm.py's SYSTEM_PROMPT regardless of what this misses.
    """
    text = _CURRENCY_PATTERN.sub("[price]", text)
    text = _EURO_PREFIX_PATTERN.sub("[price]", text)
    text = _TIME_PATTERN.sub("[price]", text)
    return text


def _is_substantial(text: str) -> bool:
    stripped = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return len(stripped) >= MIN_TURN_LENGTH


def _load_thread_messages(thread_dir: Path) -> list[dict]:
    """Merge every message_*.json in a thread and sort by timestamp.

    A handful of long threads are split across multiple numbered files by
    Meta's exporter; sorting after merging is what makes turn order correct
    for those threads.
    """
    messages: list[dict] = []
    for file_path in sorted(thread_dir.glob("message_*.json")):
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        messages.extend(raw.get("messages", []))
    messages.sort(key=lambda m: m["timestamp_ms"])
    return messages


def _collapse_turns(messages: list[dict]) -> list[tuple[str, str]]:
    """Return (role, text) turns, collapsing consecutive same-role messages.

    role is "studio" for the studio's own account, "customer" for everyone
    else. Messages with no text content (photo/attachment-only sends) are
    dropped rather than represented as placeholders.
    """
    turns: list[tuple[str, str]] = []
    for message in messages:
        content = message.get("content")
        if not content:
            continue
        sender = fix_mojibake(message.get("sender_name", ""))
        role = "studio" if sender == STUDIO_NAME else "customer"
        text = fix_mojibake(content)
        if turns and turns[-1][0] == role:
            turns[-1] = (role, turns[-1][1] + "\n" + text)
        else:
            turns.append((role, text))
    return turns


def extract_pairs(thread_id: str, messages: list[dict]) -> list[Pair]:
    """Emit one Pair for every customer turn immediately followed by a
    studio turn. A thread with no customer reply (e.g. a studio-only opening
    message) yields nothing."""
    turns = _collapse_turns(messages)
    pairs: list[Pair] = []
    for i in range(len(turns) - 1):
        role, text = turns[i]
        next_role, next_text = turns[i + 1]
        if role != "customer" or next_role != "studio":
            continue
        if not _is_substantial(text) or not _is_substantial(next_text):
            continue
        pairs.append(
            Pair(
                thread_id=thread_id,
                customer=text,
                studio_reply_scrubbed=scrub_pricing(next_text),
            )
        )
    return pairs


def dedupe_pairs(pairs: list[Pair]) -> list[Pair]:
    """Drop pairs whose (scrubbed) studio reply already appeared earlier.

    The studio's opening greeting template repeats near-verbatim across
    thousands of threads; without this, it would dominate the corpus and
    every retrieval.
    """
    seen: set[str] = set()
    deduped: list[Pair] = []
    for pair in pairs:
        if pair.studio_reply_scrubbed in seen:
            continue
        seen.add(pair.studio_reply_scrubbed)
        deduped.append(pair)
    return deduped


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_pairs: list[Pair] = []
    for thread_dir in sorted(INBOX_DIR.iterdir()):
        if not thread_dir.is_dir():
            continue
        messages = _load_thread_messages(thread_dir)
        all_pairs.extend(extract_pairs(thread_dir.name, messages))

    deduped = dedupe_pairs(all_pairs)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for pair in deduped:
            f.write(
                json.dumps(
                    {
                        "thread_id": pair.thread_id,
                        "customer": pair.customer,
                        "studio_reply_scrubbed": pair.studio_reply_scrubbed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    removed = len(all_pairs) - len(deduped)
    print(
        f"Wrote {len(deduped)} pairs from {len(all_pairs)} extracted "
        f"({removed} duplicate boilerplate replies removed) to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_rag_extract.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Run the full test suite to check for regressions**

Run: `.venv/Scripts/python -m pytest -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/__init__.py scripts/rag_extract.py tests/test_rag_extract.py
git commit -m "feat(scripts): extract customer/studio Q&A pairs from inbox/ export"
```

---

### Task 6: Offline index build script (`scripts/rag_build_index.py`)

**Files:**
- Create: `scripts/rag_build_index.py`
- Test: `tests/test_rag_build_index.py`

**Interfaces:**
- Produces (used only by a human running the script): `build_index(entries:
  list[dict], api_key: str) -> list[dict]`; `main() -> None`;
  `BATCH_SIZE: int` module constant. Output shape matches what
  `app/rag.py`'s `_load_index` reads: `[{"question": str, "reply": str,
  "embedding": list[float]}, ...]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rag_build_index.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_rag_build_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.rag_build_index'`

- [ ] **Step 3: Write `scripts/rag_build_index.py`**

```python
"""Embed the approved RAG corpus and write the runtime index.

Usage, after reviewing data/rag_corpus_review.jsonl and saving your edits to
data/rag_corpus_approved.jsonl:

    VOYAGE_API_KEY=... python -m scripts.rag_build_index

Writes data/rag_index.json, which app/rag.py loads at runtime. See
docs/superpowers/specs/2026-08-12-rag-style-retrieval-design.md.
"""
import json
import os
from pathlib import Path

import httpx

VOYAGE_ENDPOINT = "https://api.voyageai.com/v1/embeddings"
VOYAGE_MODEL = os.environ.get("VOYAGE_MODEL", "voyage-3.5")
BATCH_SIZE = 128
REQUEST_TIMEOUT_SECONDS = 30.0

APPROVED_PATH = Path("data/rag_corpus_approved.jsonl")
INDEX_PATH = Path("data/rag_index.json")


def _read_approved(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _embed_batch(texts: list[str], api_key: str) -> list[list[float]]:
    response = httpx.post(
        VOYAGE_ENDPOINT,
        json={"input": texts, "model": VOYAGE_MODEL, "input_type": "document"},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["data"]]


def build_index(entries: list[dict], api_key: str) -> list[dict]:
    indexed: list[dict] = []
    for start in range(0, len(entries), BATCH_SIZE):
        batch = entries[start : start + BATCH_SIZE]
        embeddings = _embed_batch([entry["customer"] for entry in batch], api_key)
        for entry, embedding in zip(batch, embeddings):
            indexed.append(
                {
                    "question": entry["customer"],
                    "reply": entry["studio_reply_scrubbed"],
                    "embedding": embedding,
                }
            )
    return indexed


def main() -> None:
    api_key = os.environ["VOYAGE_API_KEY"]
    entries = _read_approved(APPROVED_PATH)
    indexed = build_index(entries, api_key)
    INDEX_PATH.write_text(json.dumps(indexed, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(indexed)} embedded examples to {INDEX_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_rag_build_index.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `.venv/Scripts/python -m pytest -v`
Expected: PASS (every test in the repo, including the new RAG ones)

- [ ] **Step 6: Commit**

```bash
git add scripts/rag_build_index.py tests/test_rag_build_index.py
git commit -m "feat(scripts): build the embedded RAG index from the approved corpus"
```

---

### Task 7: Container wiring, config surface, and docs

**Files:**
- Modify: `compose.yaml`
- Modify: `Dockerfile`
- Modify: `.env.example`
- Modify: `.gitignore`
- Modify: `README.md`

No new tests — this task wires up already-tested code for deployment and
documents the offline workflow a human runs. Verification is manual (build
and boot the image, confirm nothing broke).

- [ ] **Step 1: Add the bind mount to `compose.yaml`**

```yaml
services:
  api:
    build: .
    env_file: .env
    environment:
      PORT: "${PORT:-3000}"
    ports:
      - "${PORT:-3000}:${PORT:-3000}"
    volumes:
      - history:/srv/data
      - ./data/rag_index.json:/srv/data/rag_index.json:ro
    restart: unless-stopped
```

(Only the new `- ./data/rag_index.json:/srv/data/rag_index.json:ro` line is
added under `volumes`; the `tunnel` service and everything else in the file
is unchanged.)

- [ ] **Step 2: Set the container's default index path in `Dockerfile`**

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=3000 \
    DB_PATH=/srv/data/history.db \
    RAG_INDEX_PATH=/srv/data/rag_index.json
```

- [ ] **Step 3: Document the new variables in `.env.example`**

Add after the `ASSISTANT_ADMIN_KEY` line:

```
# Optional — enables style retrieval from the studio's past DM history (see
# docs/superpowers/specs/2026-08-12-rag-style-retrieval-design.md). Leave
# VOYAGE_API_KEY blank to keep this off; the assistant behaves exactly as it
# does without it. RAG_INDEX_PATH is set to /srv/data/rag_index.json in the
# image, matching DB_PATH's pattern.
VOYAGE_API_KEY=
VOYAGE_MODEL=voyage-3.5
RAG_TOP_K=3
```

- [ ] **Step 4: Ignore the derived corpus/index files**

Add to `.gitignore` (after the existing `inbox/` line):

```
data/rag_corpus_review.jsonl
data/rag_corpus_approved.jsonl
data/rag_index.json
```

- [ ] **Step 5: Document the offline workflow in `README.md`**

Add a new section after "## Conversation storage":

```markdown
## Style retrieval from past DMs (RAG)

The assistant can draw on the studio's own past Instagram DM replies (a
Meta data export placed at `inbox/` in the project root, never committed) to
match its phrasing, without ever exposing an old price or booking time. This
is optional — leave `VOYAGE_API_KEY` unset and the assistant behaves exactly
as it does without it.

To build or refresh the corpus:

```bash
# 1. Extract candidate (question, reply) pairs from inbox/
.venv/Scripts/python -m scripts.rag_extract
# writes data/rag_corpus_review.jsonl

# 2. Review it. Copy your edited version to data/rag_corpus_approved.jsonl —
#    only what you approve here ever reaches the model.
cp data/rag_corpus_review.jsonl data/rag_corpus_approved.jsonl
# ... edit data/rag_corpus_approved.jsonl by hand ...

# 3. Embed the approved corpus via Voyage AI and write the runtime index
VOYAGE_API_KEY=... .venv/Scripts/python -m scripts.rag_build_index
# writes data/rag_index.json
```

`compose.yaml` bind-mounts `data/rag_index.json` read-only into the `api`
container. Refreshing the corpus means re-running steps 1–3 and restarting
the container (`docker compose restart api`) — no image rebuild needed. If
`data/rag_index.json` doesn't exist yet, the assistant simply runs without
style examples until you build one.
```

- [ ] **Step 6: Rebuild the image and verify it still boots**

Run: `docker compose up -d --build api`
Expected: container reaches `healthy` status, same as before this plan.

Run: `docker compose exec api python -c "import numpy; print(numpy.__version__)"`
Expected: prints the pinned numpy version, confirming it installed in the image.

Run: `docker images ai-instagram-assistant-api --format "{{.Size}}"`
Expected: comfortably under 300MB (numpy adds roughly 15-20MB).

- [ ] **Step 7: Commit**

```bash
git add compose.yaml Dockerfile .env.example .gitignore README.md
git commit -m "docs+ops: wire the RAG index into the container and document the offline pipeline"
```
