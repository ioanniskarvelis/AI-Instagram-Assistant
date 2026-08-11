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
