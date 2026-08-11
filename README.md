# Tattoo Studio Instagram Assistant

Replies to Instagram DMs for the studio using Claude, remembering the last
20 messages of each conversation. Tattoo quoting (via Telegram Q&A with the
owner) and calendar booking arrive in later slices.

## Running locally

```bash
cp .env.example .env   # then fill in IG_USER_ACCESS_TOKEN, IG_APP_SECRET, and CLOUDFLARE_TUNNEL_TOKEN
docker compose --profile dev up --build
```

`CLOUDFLARE_TUNNEL_TOKEN` comes from a named Cloudflare Tunnel, set up once:

1. In the [Cloudflare Zero Trust dashboard](https://one.dash.cloudflare.com/),
   go to **Networks > Tunnels > Create a tunnel**, choose **Docker**, and copy
   the token from the generated run command (the `--token ...` value) into
   `CLOUDFLARE_TUNNEL_TOKEN`.
2. On the tunnel's **Public Hostname** tab, add a hostname (e.g.
   `webhook.yourdomain.com`) pointing to service `HTTP` → `api:3000`. This
   requires a domain on your Cloudflare account.

That hostname is permanent — it survives container restarts, so the Meta app
dashboard only needs to be configured once:

- Callback URL: `https://<your hostname>/webhook`
- Verify token: the value of `IG_VERIFY_TOKEN` in your `.env`

Subscribe to the `messages` field, then DM the studio account from a different
Instagram account. You should receive `CANNED_REPLY` back.

## Running the tests

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest -v
```

## Configuration

See `.env.example` for every supported variable and its default.

## Conversation storage

Conversations are stored in SQLite, in a Docker named volume mounted at
`/srv/data`. Messages are deleted automatically 20 days after they are written;
a sweep runs at startup and every six hours thereafter.

Two independent settings control this:

- `HISTORY_RETENTION_DAYS` — how long a message survives on disk
- `HISTORY_WINDOW_MESSAGES` — how many recent turns are sent to the model

Customer message text is stored in plaintext for the retention period. If that
matters for your deployment, encrypt the volume at rest.

To count or clear stored conversations:

```bash
docker compose exec api python -c "import sqlite3; print(sqlite3.connect('/srv/data/history.db').execute('SELECT COUNT(*) FROM messages').fetchone()[0])"
docker compose down -v   # removes the volume and all stored conversations
```
