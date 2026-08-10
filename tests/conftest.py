import hashlib
import hmac

import pytest
from fastapi.testclient import TestClient

VERIFY_TOKEN = "test-verify-token"
ACCESS_TOKEN = "test-access-token"
APP_SECRET = "test-app-secret"
CANNED_REPLY = "Test reply"


def sign(body: bytes) -> str:
    """Build the X-Hub-Signature-256 header value Meta would send."""
    digest = hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("IG_VERIFY_TOKEN", VERIFY_TOKEN)
    monkeypatch.setenv("IG_USER_ACCESS_TOKEN", ACCESS_TOKEN)
    monkeypatch.setenv("IG_APP_SECRET", APP_SECRET)
    monkeypatch.setenv("CANNED_REPLY", CANNED_REPLY)

    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(env):
    from app.main import create_app

    return TestClient(create_app())
