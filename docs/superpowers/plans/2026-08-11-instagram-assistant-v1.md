# Instagram Assistant v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a containerized FastAPI service that receives Instagram DM webhooks, verifies Meta's signature, and replies with fixed text — proving the full loop end to end.

**Architecture:** A thin `app/` package with one responsibility per module: `config` validates environment at startup, `schemas` models the nested Instagram payload, `instagram` is the only module that knows the Graph API exists, `webhook` holds the two routes, `main` assembles the app. The container runs uvicorn as a non-root user; a `cloudflared` quick tunnel under a compose profile gives Meta a public HTTPS URL during development.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, httpx, pydantic-settings. Tests with pytest + respx. Docker + Docker Compose.

**Spec:** `docs/superpowers/specs/2026-08-11-instagram-assistant-v1-design.md`

## Global Constraints

- Python 3.12.
- Runtime dependencies are exactly: `fastapi`, `uvicorn[standard]`, `httpx`, `pydantic-settings`. Adding any other runtime dependency is out of scope.
- Test-only dependencies live in `requirements-dev.txt` and must never be installed into the image.
- Logging goes to **stdout only**. Never open a log file.
- Environment variable names are exactly: `IG_VERIFY_TOKEN`, `IG_USER_ACCESS_TOKEN`, `IG_APP_SECRET`, `IG_API_VERSION`, `PORT`, `CANNED_REPLY`, `LOG_LEVEL`.
- `IG_API_VERSION` defaults to `v22.0`. `PORT` defaults to `3000`. `LOG_LEVEL` defaults to `INFO`.
- `CANNED_REPLY` defaults to exactly: `Γεια σου! Ελάβαμε το μήνυμά σου και θα σου απαντήσουμε σύντομα.`
- `POST /webhook` returns `200` to Meta even when sending a reply fails, so Meta does not retry-storm.
- The HMAC signature is verified against the **raw** request body, before any parsing.
- The container runs as a non-root user. Final image must be under 300MB.
- v1 stores nothing. No database, no Redis, no LLM, no calendar.

---

### Task 1: Repo reset, config, and health endpoint

Clears the old codebase out of the index, lays down the package skeleton, and delivers a running app with validated configuration.

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `.gitignore`
- Create: `app/__init__.py`, `app/config.py`, `app/main.py`
- Test: `tests/__init__.py`, `tests/conftest.py`, `tests/test_config.py`, `tests/test_health.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `app.config.Settings` — pydantic-settings model with fields `ig_verify_token: str`, `ig_user_access_token: str`, `ig_app_secret: str`, `ig_api_version: str`, `port: int`, `canned_reply: str`, `log_level: str`
  - `app.config.get_settings() -> Settings` — `lru_cache`d; tests call `get_settings.cache_clear()`
  - `app.main.create_app() -> FastAPI` — app factory
  - `app.main.app` — module-level instance for uvicorn
  - `tests.conftest` fixtures `client` (a `TestClient`) and `env`, plus constants `VERIFY_TOKEN`, `ACCESS_TOKEN`, `APP_SECRET`, `CANNED_REPLY` and helper `sign(body: bytes) -> str`

- [ ] **Step 1: Stage removal of the old codebase**

The previous implementation is already deleted from disk but still tracked. Staging the deletions makes the reset explicit.

The old Greek prompt files are valuable reference material for later slices. They remain recoverable from history — for example `git show b4e26cc:prompts/pricing.txt` — so nothing is lost by removing them from the working tree.

```bash
git add -A
git status --short
```

Expected: deletions of `app.py`, `calendar_functions.py`, `prompts/*`, `requirements.txt`, `privacy_policy.html`, `terms.html`, `README.md`, `LICENSE`, `.gitattributes`, `.gitignore` are staged. Do not commit yet.

- [ ] **Step 2: Create dependency and tooling files**

`requirements.txt`:
```
fastapi==0.115.6
uvicorn[standard]==0.34.0
httpx==0.28.1
pydantic-settings==2.7.1
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest==8.3.4
pytest-asyncio==0.25.2
respx==0.22.0
```

`pytest.ini`:
```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

`.gitignore`:
```
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
.env
.env.*
!.env.example
.remember/
.DS_Store
Thumbs.db
.idea/
.vscode/
```

- [ ] **Step 3: Install dependencies**

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
```

On PowerShell the interpreter path is `.venv\Scripts\python.exe`. All later `pytest` invocations assume this virtualenv is active.

- [ ] **Step 4: Write the failing config test**

`tests/__init__.py` is empty. `tests/conftest.py`:

```python
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
```

`tests/test_config.py`:

```python
import pytest
from pydantic import ValidationError


def test_settings_read_from_environment():
    from app.config import get_settings

    settings = get_settings()
    assert settings.ig_verify_token == "test-verify-token"
    assert settings.ig_app_secret == "test-app-secret"


def test_settings_apply_documented_defaults():
    from app.config import get_settings

    settings = get_settings()
    assert settings.ig_api_version == "v22.0"
    assert settings.port == 3000
    assert settings.log_level == "INFO"


def test_missing_required_setting_raises(monkeypatch):
    from app.config import Settings, get_settings

    monkeypatch.delenv("IG_APP_SECRET", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 6: Implement the config module**

`app/__init__.py` is empty. `app/config.py`:

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CANNED_REPLY = (
    "Γεια σου! Ελάβαμε το μήνυμά σου και θα σου απαντήσουμε σύντομα."
)


class Settings(BaseSettings):
    """Application configuration, validated once at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ig_verify_token: str
    ig_user_access_token: str
    ig_app_secret: str
    ig_api_version: str = "v22.0"
    port: int = 3000
    canned_reply: str = DEFAULT_CANNED_REPLY
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 7: Run the config test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 8: Write the failing health test**

`tests/test_health.py`:

```python
def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 9: Run it to verify it fails**

Run: `pytest tests/test_health.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 10: Implement the app factory**

`app/main.py`:

```python
import logging

from fastapi import FastAPI

from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app = FastAPI(title="Tattoo Studio Instagram Assistant", version="1.0.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
```

Note: `logging.basicConfig` with no handler argument writes to stderr, which the container runtime collects. No file handler is ever added.

- [ ] **Step 11: Run the full suite**

Run: `pytest -v`
Expected: PASS (4 tests)

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "feat: reset repo and add config plus health endpoint"
```

---

### Task 2: Instagram webhook payload schemas

Models the nested payload so the route handler never inspects raw dictionaries.

**Files:**
- Create: `app/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `app.schemas.WebhookPayload` — pydantic model with `object: str` and `entry: list[Entry]`
  - `app.schemas.WebhookPayload.replyable_messages() -> list[tuple[str, str]]` — returns `(sender_id, text)` pairs for inbound text messages only
  - Supporting models `Participant`, `Message`, `MessagingEvent`, `Entry`

- [ ] **Step 1: Write the failing schema tests**

`tests/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError


def _payload(message: dict) -> dict:
    return {
        "object": "instagram",
        "entry": [
            {
                "id": "17841400000000000",
                "time": 1723000000,
                "messaging": [
                    {
                        "sender": {"id": "SENDER_1"},
                        "recipient": {"id": "STUDIO"},
                        "timestamp": 1723000000,
                        "message": message,
                    }
                ],
            }
        ],
    }


def test_text_message_is_replyable():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        _payload({"mid": "m1", "text": "Γεια σας, πόσο κοστίζει;"})
    )
    assert payload.replyable_messages() == [
        ("SENDER_1", "Γεια σας, πόσο κοστίζει;")
    ]


def test_echo_message_is_skipped():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        _payload({"mid": "m2", "text": "our own reply", "is_echo": True})
    )
    assert payload.replyable_messages() == []


def test_message_without_text_is_skipped():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        _payload({"mid": "m3", "attachments": [{"type": "image"}]})
    )
    assert payload.replyable_messages() == []


def test_event_without_message_is_skipped():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        {
            "object": "instagram",
            "entry": [
                {
                    "id": "E1",
                    "messaging": [
                        {
                            "sender": {"id": "SENDER_1"},
                            "recipient": {"id": "STUDIO"},
                            "read": {"mid": "m4"},
                        }
                    ],
                }
            ],
        }
    )
    assert payload.replyable_messages() == []


def test_multiple_entries_and_events_are_all_collected():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        {
            "object": "instagram",
            "entry": [
                {
                    "id": "E1",
                    "messaging": [
                        {
                            "sender": {"id": "A"},
                            "recipient": {"id": "STUDIO"},
                            "message": {"mid": "m1", "text": "one"},
                        },
                        {
                            "sender": {"id": "B"},
                            "recipient": {"id": "STUDIO"},
                            "message": {"mid": "m2", "text": "two"},
                        },
                    ],
                },
                {
                    "id": "E2",
                    "messaging": [
                        {
                            "sender": {"id": "C"},
                            "recipient": {"id": "STUDIO"},
                            "message": {"mid": "m3", "text": "three"},
                        }
                    ],
                },
            ],
        }
    )
    assert payload.replyable_messages() == [
        ("A", "one"),
        ("B", "two"),
        ("C", "three"),
    ]


def test_payload_missing_object_is_rejected():
    from app.schemas import WebhookPayload

    with pytest.raises(ValidationError):
        WebhookPayload.model_validate({"entry": []})
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.schemas'`

- [ ] **Step 3: Implement the schemas**

`app/schemas.py`:

```python
from pydantic import BaseModel, ConfigDict, Field


class Participant(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mid: str | None = None
    text: str | None = None
    is_echo: bool = False


class MessagingEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sender: Participant
    recipient: Participant
    message: Message | None = None


class Entry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    messaging: list[MessagingEvent] = Field(default_factory=list)


class WebhookPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    object: str
    entry: list[Entry] = Field(default_factory=list)

    def replyable_messages(self) -> list[tuple[str, str]]:
        """Return (sender_id, text) for inbound text messages worth replying to.

        Skips echoes — the studio's own outbound replies are delivered back to
        this webhook, and replying to them makes the assistant answer itself in
        a loop. Also skips events with no text: reactions, attachments, read
        receipts and delivery receipts all arrive on this same endpoint.
        """
        replyable: list[tuple[str, str]] = []
        for entry in self.entry:
            for event in entry.messaging:
                message = event.message
                if message is None or message.is_echo or not message.text:
                    continue
                replyable.append((event.sender.id, message.text))
        return replyable
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py tests/test_schemas.py
git commit -m "feat: model Instagram webhook payload with echo filtering"
```

---

### Task 3: Instagram Graph API client

The only module that knows how replies are sent.

**Files:**
- Create: `app/instagram.py`
- Test: `tests/test_instagram.py`

**Interfaces:**
- Consumes: `app.config.get_settings`
- Produces:
  - `app.instagram.send_text(recipient_id: str, text: str) -> bool` — async; returns `True` on success, `False` on a Graph API error or transport failure. Never raises.
  - `app.instagram.GRAPH_BASE` — `"https://graph.instagram.com"`

- [ ] **Step 1: Write the failing client tests**

`tests/test_instagram.py`:

```python
import json

import httpx
import respx

from tests.conftest import ACCESS_TOKEN

ENDPOINT = "https://graph.instagram.com/v22.0/me/messages"

# respx patches httpcore, which the app's httpx.AsyncClient goes through.
# TestClient's ASGITransport bypasses httpcore entirely, so requests to the
# app under test are never intercepted — only outbound Graph API calls are.


@respx.mock
async def test_send_text_posts_expected_payload():
    from app.instagram import send_text

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    assert await send_text("SENDER_1", "hello") is True
    assert route.called

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {
        "recipient": {"id": "SENDER_1"},
        "message": {"text": "hello"},
    }
    assert request.headers["Authorization"] == f"Bearer {ACCESS_TOKEN}"


@respx.mock
async def test_send_text_returns_false_on_api_error():
    from app.instagram import send_text

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad token"}})
    )

    assert await send_text("SENDER_1", "hello") is False


@respx.mock
async def test_send_text_returns_false_on_transport_error():
    from app.instagram import send_text

    respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("boom"))

    assert await send_text("SENDER_1", "hello") is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_instagram.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.instagram'`

- [ ] **Step 3: Implement the client**

`app/instagram.py`:

```python
import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.instagram.com"
REQUEST_TIMEOUT_SECONDS = 10.0


async def send_text(recipient_id: str, text: str) -> bool:
    """Send a plain-text DM. Returns True on success, False on failure.

    Never raises: the webhook must return 200 to Meta regardless of whether
    the reply went out, so failures are logged and reported as a bool.
    """
    settings = get_settings()
    url = f"{GRAPH_BASE}/{settings.ig_api_version}/me/messages"
    payload = {"recipient": {"id": recipient_id}, "message": {"text": text}}
    headers = {"Authorization": f"Bearer {settings.ig_user_access_token}"}

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError:
        logger.exception("Transport error sending reply to %s", recipient_id)
        return False

    if response.status_code >= 400:
        logger.error(
            "Graph API rejected reply to %s: %s %s",
            recipient_id,
            response.status_code,
            response.text,
        )
        return False

    return True
```

The access token goes in the `Authorization` header rather than a query string, keeping it out of URLs and any logs that record them.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_instagram.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/instagram.py tests/test_instagram.py
git commit -m "feat: add Instagram Graph API send_text client"
```

---

### Task 4: Webhook verification endpoint

Meta's subscription handshake. Must work before any message can arrive.

**Files:**
- Create: `app/webhook.py`
- Modify: `app/main.py` (mount the router)
- Test: `tests/test_webhook_verify.py`

**Interfaces:**
- Consumes: `app.config.get_settings`
- Produces:
  - `app.webhook.router` — an `APIRouter` carrying `GET /webhook`. Task 5 adds `POST /webhook` to the same router.

- [ ] **Step 1: Write the failing verification tests**

`tests/test_webhook_verify.py`:

```python
from tests.conftest import VERIFY_TOKEN


def test_verification_returns_challenge_on_correct_token(client):
    response = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": VERIFY_TOKEN,
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 200
    assert response.text == "1158201444"


def test_verification_rejects_wrong_token(client):
    response = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong-token",
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 403


def test_verification_rejects_wrong_mode(client):
    response = client.get(
        "/webhook",
        params={
            "hub.mode": "unsubscribe",
            "hub.verify_token": VERIFY_TOKEN,
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 403


def test_verification_rejects_missing_params(client):
    response = client.get("/webhook")
    assert response.status_code == 403
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_webhook_verify.py -v`
Expected: FAIL — all four return 404, since no `/webhook` route exists yet

- [ ] **Step 3: Implement the verification route**

`app/webhook.py`:

```python
import hmac
import logging

from fastapi import APIRouter, Request, Response, status

from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/webhook")
async def verify(request: Request) -> Response:
    """Answer Meta's subscription handshake."""
    settings = get_settings()
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge", "")

    if (
        mode == "subscribe"
        and token is not None
        and hmac.compare_digest(token, settings.ig_verify_token)
    ):
        return Response(content=challenge, media_type="text/plain")

    logger.warning("Webhook verification failed (mode=%s)", mode)
    return Response(status_code=status.HTTP_403_FORBIDDEN)
```

`hmac.compare_digest` is used instead of `==` to compare the token in constant time.

- [ ] **Step 4: Mount the router**

In `app/main.py`, add the import below the existing `from app.config import get_settings` line:

```python
from app.webhook import router
```

and inside `create_app`, immediately after the `app = FastAPI(...)` line, add:

```python
    app.include_router(router)
```

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/test_webhook_verify.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Run the full suite**

Run: `pytest -v`
Expected: PASS (17 tests)

- [ ] **Step 7: Commit**

```bash
git add app/webhook.py app/main.py tests/test_webhook_verify.py
git commit -m "feat: add webhook verification endpoint"
```

---

### Task 5: Webhook message receipt

The core of v1: verify the signature, filter echoes, send the canned reply.

**Files:**
- Modify: `app/webhook.py` (add the POST route)
- Test: `tests/test_webhook_receive.py`

**Interfaces:**
- Consumes: `app.schemas.WebhookPayload`, `app.instagram.send_text`, `app.config.get_settings`, `tests.conftest.sign`
- Produces: `POST /webhook` on the existing `app.webhook.router`; module-level helper `app.webhook._signature_valid(raw_body: bytes, header: str | None, app_secret: str) -> bool`

- [ ] **Step 1: Write the failing receipt tests**

`tests/test_webhook_receive.py`:

```python
import json

import httpx
import respx

from tests.conftest import CANNED_REPLY, sign

ENDPOINT = "https://graph.instagram.com/v22.0/me/messages"


def _body(message: dict) -> bytes:
    payload = {
        "object": "instagram",
        "entry": [
            {
                "id": "STUDIO",
                "time": 1723000000,
                "messaging": [
                    {
                        "sender": {"id": "SENDER_1"},
                        "recipient": {"id": "STUDIO"},
                        "timestamp": 1723000000,
                        "message": message,
                    }
                ],
            }
        ],
    }
    return json.dumps(payload).encode()


def _post(client, body: bytes, signature: str | None = None):
    headers = {"Content-Type": "application/json"}
    headers["X-Hub-Signature-256"] = sign(body) if signature is None else signature
    return client.post("/webhook", content=body, headers=headers)


@respx.mock
def test_text_message_triggers_one_reply(client):
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    body = _body({"mid": "m1", "text": "Γεια σας"})
    response = _post(client, body)

    assert response.status_code == 200
    assert route.call_count == 1

    sent = json.loads(route.calls.last.request.content)
    assert sent["recipient"]["id"] == "SENDER_1"
    assert sent["message"]["text"] == CANNED_REPLY


@respx.mock
def test_echo_message_triggers_no_reply(client):
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={}))

    body = _body({"mid": "m2", "text": "our own reply", "is_echo": True})
    response = _post(client, body)

    assert response.status_code == 200
    assert route.call_count == 0


@respx.mock
def test_invalid_signature_is_rejected(client):
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={}))

    body = _body({"mid": "m3", "text": "hello"})
    response = _post(client, body, signature="sha256=deadbeef")

    assert response.status_code == 403
    assert route.call_count == 0


@respx.mock
def test_missing_signature_header_is_rejected(client):
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={}))

    body = _body({"mid": "m4", "text": "hello"})
    response = client.post(
        "/webhook", content=body, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 403
    assert route.call_count == 0


@respx.mock
def test_malformed_payload_returns_400(client):
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={}))

    body = json.dumps({"not": "a webhook"}).encode()
    response = _post(client, body)

    assert response.status_code == 400
    assert route.call_count == 0


@respx.mock
def test_send_failure_still_returns_200(client):
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(500, json={"error": "server error"})
    )

    body = _body({"mid": "m5", "text": "hello"})
    response = _post(client, body)

    assert response.status_code == 200
```

The final test encodes the constraint that matters most operationally: a failed send must not be reported to Meta as an error, or Meta retries a request that will fail identically.

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_webhook_receive.py -v`
Expected: FAIL — all six return 405 Method Not Allowed, since only `GET /webhook` exists

- [ ] **Step 3: Implement signature verification and the POST route**

In `app/webhook.py`, extend the imports at the top to:

```python
import hashlib
import hmac
import logging

from fastapi import APIRouter, Request, Response, status
from pydantic import ValidationError

from app.config import get_settings
from app.instagram import send_text
from app.schemas import WebhookPayload
```

Then append to the end of the file:

```python
def _signature_valid(raw_body: bytes, header: str | None, app_secret: str) -> bool:
    """Check Meta's X-Hub-Signature-256 against the raw request body."""
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.removeprefix("sha256="))


@router.post("/webhook")
async def receive(request: Request) -> Response:
    """Receive DM events and reply.

    The signature is checked against the raw body before parsing. Validating a
    re-serialised body would compare a different byte sequence than Meta signed,
    producing a check that silently always fails or, worse, is trivially bypassed.
    """
    settings = get_settings()
    raw_body = await request.body()

    if not _signature_valid(
        raw_body, request.headers.get("X-Hub-Signature-256"), settings.ig_app_secret
    ):
        logger.warning("Rejected webhook delivery with invalid signature")
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    try:
        payload = WebhookPayload.model_validate_json(raw_body)
    except ValidationError as exc:
        logger.warning("Malformed webhook payload: %s", exc)
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

    for sender_id, text in payload.replyable_messages():
        logger.info("Replying to %s (received %d chars)", sender_id, len(text))
        await send_text(sender_id, settings.canned_reply)

    return Response(status_code=status.HTTP_200_OK)
```

The inbound message text is logged only as a length, not as content — these are customers' private messages.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_webhook_receive.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: PASS (23 tests)

- [ ] **Step 6: Commit**

```bash
git add app/webhook.py tests/test_webhook_receive.py
git commit -m "feat: receive signed DM webhooks and send canned reply"
```

---

### Task 6: Containerization

Packages the service and gives it a public URL for testing against real Instagram traffic.

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `compose.yaml`, `.env.example`, `README.md`

**Interfaces:**
- Consumes: `app.main:app` (the uvicorn target), every environment variable from Task 1
- Produces: a runnable image and `docker compose` services `api` and `tunnel`

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=3000

WORKDIR /srv

# Dependencies install as their own layer so code edits do not invalidate them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser app ./app
USER appuser

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import os, sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '3000') + '/health').status == 200 else 1)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
```

`exec` in the CMD makes uvicorn PID 1, so it receives `SIGTERM` on `docker stop` and shuts down cleanly instead of being killed after the timeout.

- [ ] **Step 2: Write `.dockerignore`**

```
.git
.gitignore
.gitattributes
.env
.env.*
!.env.example
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
tests/
docs/
.remember/
README.md
requirements-dev.txt
pytest.ini
```

- [ ] **Step 3: Write `compose.yaml`**

```yaml
services:
  api:
    build: .
    env_file: .env
    environment:
      PORT: "${PORT:-3000}"
    ports:
      - "${PORT:-3000}:${PORT:-3000}"
    restart: unless-stopped

  tunnel:
    image: cloudflare/cloudflared:latest
    profiles: ["dev"]
    command: tunnel --no-autoupdate --url http://api:${PORT:-3000}
    depends_on:
      - api
    restart: unless-stopped
```

The `dev` profile keeps the tunnel out of a plain `docker compose up`, so a production deployment starts only `api`.

- [ ] **Step 4: Write `.env.example`**

```
# Meta webhook handshake — any string you choose; paste the same value into
# the Meta app's webhook configuration.
IG_VERIFY_TOKEN=choose-a-random-string

# Instagram user access token with instagram_business_manage_messages.
IG_USER_ACCESS_TOKEN=

# App secret from the Meta app dashboard, used to verify payload signatures.
IG_APP_SECRET=

# Optional — defaults shown.
IG_API_VERSION=v22.0
PORT=3000
LOG_LEVEL=INFO
CANNED_REPLY=Γεια σου! Ελάβαμε το μήνυμά σου και θα σου απαντήσουμε σύντομα.
```

- [ ] **Step 5: Write `README.md`**

````markdown
# Tattoo Studio Instagram Assistant

Replies to Instagram DMs for the studio. v1 sends a fixed acknowledgement;
LLM replies, tattoo quoting and calendar booking arrive in later slices.

## Running locally

```bash
cp .env.example .env   # then fill in IG_USER_ACCESS_TOKEN and IG_APP_SECRET
docker compose --profile dev up --build
```

The `tunnel` service prints a public HTTPS URL, for example
`https://random-words-here.trycloudflare.com`. In the Meta app dashboard set:

- Callback URL: `<that URL>/webhook`
- Verify token: the value of `IG_VERIFY_TOKEN` in your `.env`

Subscribe to the `messages` field, then DM the studio account from a different
Instagram account. You should receive `CANNED_REPLY` back.

Quick tunnel URLs change on every restart, so the callback URL must be updated
in the Meta dashboard each time you restart the tunnel.

## Running the tests

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest -v
```

## Configuration

See `.env.example` for every supported variable and its default.
````

- [ ] **Step 6: Build the image and check its size**

```bash
docker build -t ig-assistant:v1 .
docker image inspect ig-assistant:v1 --format "{{.Size}}"
```

Expected: build succeeds; size is under 300000000 (300MB).

- [ ] **Step 7: Verify the container runs as non-root and answers health checks**

Use a throwaway env file rather than `.env`, so a real `.env` containing live
secrets is never overwritten by this check.

```bash
cp .env.example .env.check
docker run --rm -d --name ig-check --env-file .env.check -p 3000:3000 ig-assistant:v1
docker exec ig-check id -u
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/health
docker stop ig-check
rm .env.check
```

Expected: `id -u` prints `1000`, and the health check prints `200`.

The example values leave `IG_USER_ACCESS_TOKEN` and `IG_APP_SECRET` empty. Both
are typed `str` with no minimum length, so startup succeeds and `/health`
responds — this step checks packaging, not credentials.

- [ ] **Step 8: Verify compose brings up the tunnel**

```bash
docker compose --profile dev up --build -d
docker compose logs tunnel
```

Expected: the logs contain a `https://<something>.trycloudflare.com` URL.

Then tear down: `docker compose --profile dev down`

- [ ] **Step 9: Commit**

```bash
git add Dockerfile .dockerignore compose.yaml .env.example README.md
git commit -m "feat: containerize the service with a dev tunnel"
```

---

## Final Verification

Run before declaring v1 complete:

- [ ] `pytest -v` — 23 tests pass
- [ ] `docker build -t ig-assistant:v1 .` succeeds, image under 300MB
- [ ] `docker compose --profile dev up --build` starts both services and prints a tunnel URL
- [ ] Meta's webhook subscription completes against `<tunnel-url>/webhook` with your verify token
- [ ] A DM from another Instagram account receives `CANNED_REPLY`
- [ ] The assistant does not reply to its own outbound message (no reply loop)
- [ ] `git status` is clean

The last three require real Meta credentials and a live Instagram account, so
they are performed by the user, not by an automated worker.
