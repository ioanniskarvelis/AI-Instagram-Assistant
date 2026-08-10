# Instagram Assistant — v1 Design (Containerized Plumbing Slice)

**Date:** 2026-08-11
**Status:** Approved, ready for planning

## Context

The tattoo studio's Instagram DM assistant is being reimplemented from scratch. The
previous version (commit `b4e26cc`) is treated as reference material only, not as a
constraint — its working tree had already been deleted, and no code carries over.

That earlier version put webhook handling, intent classification, OpenAI vision
quoting, Google Calendar booking, Pinecone retrieval, and Redis state into a single
1,300-line `app.py`. The reimplementation is delivered in slices, each independently
runnable in Docker.

**This spec covers v1 only: the plumbing slice.** A customer sends a DM, the app
replies with fixed text, and the whole thing runs in a container. No LLM, no
database, no calendar.

## Goals

1. Receive Instagram DM webhook events and reply, proving the full loop end to end.
2. Run in Docker from day one, testable locally against real Instagram traffic.
3. Establish module boundaries that later slices extend rather than sprawl into.

## Non-Goals

Explicitly deferred to later slices:

- LLM-generated replies, intent classification, per-intent prompts
- Image-based tattoo quoting
- Google Calendar booking, rescheduling, cancellation
- Vector retrieval over pricing or conversation history
- Message batching / grace window
- Conversation persistence of any kind

## Stack Decisions

| Choice | Decision | Rationale |
|---|---|---|
| Language | Python 3.12 | LLM ecosystem, Google clients, existing familiarity |
| Framework | FastAPI + uvicorn | Workload is slow I/O (OpenAI, Graph API, vector search); async scales on less memory than sync workers. Meta wants a fast `200`, and background tasks are native. Pydantic models tame the deeply nested, inconsistent Instagram payload — the main source of sprawl in the old code. |
| HTTP client | httpx | Async, matches the framework |
| Config | pydantic-settings | Declarative validation, fails loudly at startup |
| Persistence | None in v1 | Nothing to store when the reply is fixed |

### Decisions recorded for later slices

Not implemented in v1, but agreed so later work doesn't relitigate them:

- **Embeddings via the OpenAI API, not `sentence-transformers`.** The local model
  pulled torch plus ~1GB of weights, making a ~3GB image. `text-embedding-3-small`
  through the existing OpenAI dependency keeps the image near 200MB and removes the
  model-cache volume entirely.
- **Google Calendar via a service account, not interactive OAuth.** The old
  `token.json` flow required a browser consent step and a mutable, mounted,
  refreshable credential file — the hardest thing in the system to containerize.
  Sharing the studio calendar with a service account reduces this to one injected
  JSON with no interactive step.
- **Postgres + pgvector is the preferred store over Pinecone**, since conversations,
  appointments, and customer records need relational storage anyway. Redis stays a
  candidate for the message-batching window, where TTL semantics fit well. The system
  should not end up running Redis *and* Pinecone *and* Postgres.

## Architecture

```
app/
  config.py      env loading + validation (pydantic-settings)
  schemas.py     Pydantic models for the Instagram webhook payload
  instagram.py   async Graph API client
  webhook.py     APIRouter: GET verify, POST receive
  main.py        FastAPI app, /health
tests/
Dockerfile
compose.yaml
.dockerignore
.env.example
requirements.txt
```

Runtime dependencies: `fastapi`, `uvicorn[standard]`, `httpx`, `pydantic-settings`.
Test-only dependencies live in `requirements-dev.txt`: `pytest`, `pytest-asyncio`,
`respx` (mocks httpx at the transport layer). These are not installed into the image.

### Module responsibilities

**`config.py`** — a `Settings` model read from the environment at import. Missing
required values raise at startup, not at the first customer DM.

**`schemas.py`** — models `entry[].messaging[]` with `message.is_echo` and optional
`message.text` as real fields. Echoes and non-text events (reactions, attachments,
read receipts) are filtered at this layer, keeping the handler free of nested
conditionals.

**`instagram.py`** — one function, `send_text(recipient_id, text)`, POSTing to
`graph.instagram.com/{version}/me/messages`. The only module that knows the Graph
API exists.

**`webhook.py`** — the two routes. Depends on `schemas` and `instagram`; contains no
HTTP-client or payload-shape details of its own.

**`main.py`** — builds the app, mounts the router, exposes `/health` returning `200`
with `{"status": "ok"}`. v1 has no dependencies to health-check beyond the process
being alive.

### Configuration

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `IG_VERIFY_TOKEN` | yes | — | Meta webhook handshake |
| `IG_USER_ACCESS_TOKEN` | yes | — | Authenticates reply sending |
| `IG_APP_SECRET` | yes | — | Verifies payload signatures |
| `IG_API_VERSION` | no | `v22.0` | Graph API version |
| `PORT` | no | `3000` | Listen port |
| `CANNED_REPLY` | no | `Γεια σου! Ελάβαμε το μήνυμά σου και θα σου απαντήσουμε σύντομα.` | The v1 reply text |
| `LOG_LEVEL` | no | `INFO` | Log verbosity |

## Data Flow

### Webhook verification — `GET /webhook`

Compare `hub.verify_token` against `IG_VERIFY_TOKEN`. On match return `hub.challenge`
as plain text with `200`; otherwise `403`.

### Message receipt — `POST /webhook`

1. Read the **raw** request body.
2. Verify the `X-Hub-Signature-256` HMAC against `IG_APP_SECRET`. Reject with `403`
   on mismatch. This happens *before* parsing — validating a re-serialized body would
   produce a signature check that silently does not work.
3. Parse the raw body into the Pydantic payload model.
4. For each messaging event: skip if `is_echo` is true, skip if `text` is absent.
5. Send `CANNED_REPLY` to the sender via `instagram.send_text`.
6. Return `200`.

Echo filtering is load-bearing, not defensive: the studio account's own outbound
replies are delivered back to the webhook, so without it the assistant answers
itself in a loop.

v1 sends the reply **synchronously** before returning `200`. A Graph API call is fast
enough that Meta's timeout is not at risk, and it surfaces failures immediately in the
logs while the loop is being proven. When the LLM slice lands, calls become slow and
this moves to acking first with the send in a background task — a deliberate later
change.

## Error Handling

- **Send failure:** log at ERROR with the sender ID and the Graph API response body,
  then still return `200`. Returning an error would trigger Meta's retry mechanism and
  duplicate work for a failure that retrying will not fix.
- **Signature mismatch:** `403`, logged at WARNING. The endpoint is publicly
  reachable, so unsigned traffic is expected and should not be noisy.
- **Malformed payload:** `400`, logged at WARNING with the validation error.
- **Missing configuration:** the process fails to start with an explicit message
  naming the missing variable.
- **Logging destination:** stdout only. No `app.log` or `schedule.log` files — in a
  container the runtime owns log collection. This is an intentional break from the
  previous version.

## Container Design

**Dockerfile** — `python:3.12-slim`. `requirements.txt` is copied and installed as its
own layer before application code, so dependency installs stay cached across code
edits. Runs as a non-root user. `HEALTHCHECK` polls `/health`. Starts uvicorn bound to
`0.0.0.0:${PORT}` so the image is host-agnostic. Expected size ~200MB.

**compose.yaml** — two services:

- `api` — builds from the local Dockerfile, `env_file: .env`, publishes `3000`,
  `restart: unless-stopped`.
- `tunnel` — `cloudflared` running a quick tunnel against `http://api:3000`, behind a
  `dev` profile so it starts only with `--profile dev`. Quick tunnels require no
  Cloudflare account and print the public HTTPS URL needed for Meta's webhook config.

**`.dockerignore`** excludes `.git`, `.env`, `__pycache__`, `tests/`, and `docs/`.

**`.env.example`** documents every variable in the table above, with no real secrets.

## Testing

`pytest` against FastAPI's `TestClient`, with the Graph API mocked at the `httpx`
boundary.

| Case | Expected |
|---|---|
| `GET /webhook` with correct verify token | `200`, body is the challenge |
| `GET /webhook` with wrong verify token | `403` |
| `POST /webhook`, valid signature, text message | exactly one send, correct recipient and text |
| `POST /webhook`, valid signature, `is_echo: true` | no send, `200` |
| `POST /webhook`, invalid signature | `403`, no send |
| `POST /webhook`, malformed body | `400`, no send |
| `GET /health` | `200`, `{"status": "ok"}` |

## Acceptance Criteria

v1 is done when:

1. `docker compose --profile dev up` starts the API and prints a public tunnel URL.
2. That URL plus the verify token completes Meta's webhook subscription successfully.
3. A DM sent to the studio account from another Instagram account receives
   `CANNED_REPLY` back.
4. The assistant does not reply to its own outbound messages.
5. `pytest` passes with every case above covered.
6. The built image is under 300MB and runs as a non-root user.
