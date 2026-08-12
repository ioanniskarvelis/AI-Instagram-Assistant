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
