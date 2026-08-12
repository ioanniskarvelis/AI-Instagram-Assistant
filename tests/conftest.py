import hashlib
import hmac

import pytest
from fastapi.testclient import TestClient

VERIFY_TOKEN = "test-verify-token"
ACCESS_TOKEN = "test-access-token"
APP_SECRET = "test-app-secret"
ACCOUNT_ID = "STUDIO"
CANNED_REPLY = "Test reply"
ANTHROPIC_API_KEY = "test-anthropic-key"
ADMIN_KEY = "test-admin-key"


def sign(body: bytes) -> str:
    """Build the X-Hub-Signature-256 header value Meta would send."""
    digest = hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


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
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_MODEL", "voyageai/voyage-4")
    monkeypatch.setenv("RAG_TOP_K", "3")
    monkeypatch.setenv("RAG_INDEX_PATH", str(tmp_path / "rag_index.json"))
    monkeypatch.setenv("INTENT_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("INTENT_MAX_TOKENS", "50")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "")
    # Reply generation now waits out a burst-debounce window in the
    # background before replying (see app.webhook). Zero it by default so
    # existing tests, which post one message and expect an immediate reply,
    # keep working; tests of the debounce/coalescing behavior itself override
    # this explicitly.
    monkeypatch.setenv("REPLY_DEBOUNCE_SECONDS", "0")

    from app.config import get_settings
    from app.rag import _load_index

    get_settings.cache_clear()
    _load_index.cache_clear()
    yield
    get_settings.cache_clear()
    _load_index.cache_clear()


@pytest.fixture
def client(env):
    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
