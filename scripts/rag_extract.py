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

# The studio's automated first-contact reply (a Meta "instant reply" or saved
# reply, not something Christina typed), sent verbatim across thousands of
# threads. A pair whose entire studio turn is *exactly* this text carries no
# style signal — it's canned, not organic phrasing — so it's dropped outright
# rather than merely deduped down to one instance. If real content follows it
# in the same turn (Christina typed something right after), the pair is kept.
AUTO_REPLY_TEXT = (
    "Γεια σας, έχουμε λάβει το μήνυμά σας και ευχαριστούμε που επικοινωνήσατε "
    "μαζί μας. Αν ενδιαφέρεστε για tattoo στείλτε μας φωτογραφία τι σχέδιο "
    "θέλετε και τι μέγεθος στο περίπου , για να μάθετε τιμή και για "
    "οποιαδήποτε άλλη πληροφορία . σύντομα κάποιος εκπρόσωπος μας θα "
    "επικοινωνήσει μαζι σας 😊😊\n②③①⓪ⓣⓔⓐⓜ"
)

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

# Real DM data rarely spells out a currency word: the studio typically just
# states a bare number or range ("180 me 220", "40-45 to proto") once price
# has already come up in the exchange, and gives a bare "18:00" once booking
# has come up. These signal words gate a second, more permissive pass over
# the studio's reply — see scrub_pricing. Patterns are plain (unaccented)
# stems, matched against accent-stripped text via _strip_accents: Greek
# stress accents land on different letters across inflected forms (e.g.
# "κοστίζει" vs "κόστος"), so a literal accented stem would silently miss
# half its own inflections.
_PRICE_SIGNAL_PATTERN = re.compile(
    r"τιμ|κοστ|ευρω|€|εκπτ|συνολ|καθεν|how much|cost|price|euro",
    re.IGNORECASE,
)
_BOOKING_SIGNAL_PATTERN = re.compile(
    r"ραντεβου|διαθεσιμ|ωρ[αες]|κλεισ|book|appointment|μερες"
    # Day names: a day name next to a bare time ("Παρασκευή 13:00") is a
    # booking slot even when neither exchange turn says "ραντεβού"/"ώρα"
    # outright — common when the appointment context was set earlier in
    # the thread, outside the two-turn window this script looks at.
    r"|δευτερ|τριτ|τεταρτ|πεμπτ|παρασκευ|σαββατ|κυριακ",
    re.IGNORECASE,
)
# Two digits minimum: a bare single digit ("2 σχέδια") is far more often a
# quantity than a price, so it's left alone even on a price-signal line.
_BARE_NUMBER_PATTERN = re.compile(
    r"\d{2,}(?:[.,]\d+)?(?:\s*(?:[-–]|με|έως|to)\s*\d{2,}(?:[.,]\d+)?)?"
)
_BARE_TIME_PATTERN = re.compile(r"\d{1,2}[:.]\d{2}")

# One codepoint in, one codepoint out, so the result stays the same length
# and every character stays at the same index as the original — required
# for _scrub_dates below, which matches against the stripped text but
# substitutes into the original (accented) text by reusing match spans.
_ACCENT_MAP = str.maketrans(
    "άέήίόύώΐΰϊϋΆΈΉΊΌΎΏΪΫ",
    "αεηιουωιυιυΑΕΗΙΟΥΩΙΥ",
)

# Day-of-week names are deliberately NOT scrubbed anywhere in this module:
# "Παρασκευή" is timeless vocabulary, not a stale fact, and removing it would
# cost real style for no privacy/staleness benefit. A calendar date is a
# different matter — "19 Νοεμβρίου" is exactly as stale as a clock time.
_MONTH_NAMES = (
    "ιανουαριου", "φεβρουαριου", "μαρτιου", "απριλιου", "μαιου", "ιουνιου",
    "ιουλιου", "αυγουστου", "σεπτεμβριου", "οκτωβριου", "νοεμβριου",
    "δεκεμβριου",
)
_DATE_PATTERN = re.compile(
    r"\d{1,2}\s*(?:του\s+μηνος|" + "|".join(_MONTH_NAMES) + r")",
    re.IGNORECASE,
)


def _strip_accents(text: str) -> str:
    """Fold away Greek tonos marks so a plain stem like "κοστ" matches every
    inflected form regardless of where the stress accent falls."""
    return text.translate(_ACCENT_MAP)


def _scrub_dates(text: str) -> str:
    """Replace "<day> <month name>" / "<day> του μηνός" with "[time]".

    Matches against an accent-stripped copy (month names are far more often
    typed with the accent than not, but this shouldn't depend on it) and
    reuses the match spans directly against the original text — safe only
    because _strip_accents is length- and position-preserving.
    """
    stripped = _strip_accents(text)
    pieces: list[str] = []
    last = 0
    for match in _DATE_PATTERN.finditer(stripped):
        pieces.append(text[last : match.start()])
        pieces.append("[time]")
        last = match.end()
    pieces.append(text[last:])
    return "".join(pieces)


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


def scrub_pricing(customer_text: str, studio_reply: str) -> str:
    """Replace price, booking-time, and calendar-date mentions in the
    studio's reply with a neutral placeholder — "[price]" for currency
    amounts, "[time]" for booking times/dates, so a human skimming the
    review file isn't stuck decoding a mislabeled token. Bare weekday names
    ("Παρασκευή") are deliberately left alone — see _MONTH_NAMES above.

    Currency-symbol/word-adjacent numbers are always scrubbed. Beyond that,
    a bare number or time in the reply usually only reads as a price or a
    booking slot in light of the surrounding exchange (the customer asked
    "τιμή;" and the studio just states "180-220"; the customer asked for a
    ραντεβού and the studio just states "18:00") — so `customer_text` is
    also scanned for the signal words that gate that second, more
    permissive pass.

    Heuristic, not exhaustive — the human review step (data/rag_corpus_review.jsonl)
    is the real safety net; the hard rule against quoting prices lives in
    app/llm.py's SYSTEM_PROMPT regardless of what this misses.
    """
    reply = _CURRENCY_PATTERN.sub("[price]", studio_reply)
    reply = _EURO_PREFIX_PATTERN.sub("[price]", reply)
    reply = _TIME_PATTERN.sub("[time]", reply)
    reply = _scrub_dates(reply)

    context = _strip_accents(customer_text + "\n" + reply)
    # Time first: a bare "18:00" must be consumed as one token before the
    # bare-number pass, or it fragments into two separate placeholders on
    # either side of the colon.
    if _BOOKING_SIGNAL_PATTERN.search(context):
        reply = _BARE_TIME_PATTERN.sub("[time]", reply)
    if _PRICE_SIGNAL_PATTERN.search(context):
        reply = _BARE_NUMBER_PATTERN.sub("[price]", reply)
    return reply


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
        if next_text.strip() == AUTO_REPLY_TEXT.strip():
            continue
        pairs.append(
            Pair(
                thread_id=thread_id,
                customer=text,
                studio_reply_scrubbed=scrub_pricing(text, next_text),
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
