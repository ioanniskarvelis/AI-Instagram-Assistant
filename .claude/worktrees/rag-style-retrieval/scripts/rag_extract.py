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
