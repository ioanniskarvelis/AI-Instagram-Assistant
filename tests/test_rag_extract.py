import json

from scripts.rag_extract import (
    AUTO_REPLY_TEXT,
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
    assert (
        scrub_pricing("γεια", "Θα σου κοστίσει 150€ περίπου")
        == "Θα σου κοστίσει [price] περίπου"
    )


def test_scrub_pricing_replaces_euro_word_amount():
    assert scrub_pricing("γεια", "κοστίζει 150 ευρώ") == "κοστίζει [price]"


def test_scrub_pricing_replaces_range():
    assert scrub_pricing("γεια", "κοστίζει 100-150 ευρώ") == "κοστίζει [price]"


def test_scrub_pricing_replaces_booking_time():
    assert (
        scrub_pricing("γεια", "Ελα αύριο στις 5 μ.μ.") == "Ελα αύριο στις [time]"
    )


def test_scrub_pricing_replaces_english_euro_amount():
    assert (
        scrub_pricing("γεια", "It will cost around 150 euros for that size")
        == "It will cost around [price] for that size"
    )


def test_scrub_pricing_leaves_unrelated_numbers_alone():
    assert (
        scrub_pricing("γεια", "θέλω tattoo 10 πόντους")
        == "θέλω tattoo 10 πόντους"
    )


def test_scrub_pricing_replaces_bare_range_when_customer_asked_price():
    # Real DM pattern: the studio just states a bare range, no currency word,
    # once the customer has already asked "τιμη".
    assert (
        scrub_pricing(
            "θα ηθελα να ρωτησω για μια τιμη στο περίπου",
            "Καλησπέρα στα 180 με 220 ανάλογα το μέγεθος",
        )
        == "Καλησπέρα στα [price] ανάλογα το μέγεθος"
    )


def test_scrub_pricing_replaces_bare_range_when_reply_itself_signals_price():
    # No price word from the customer, but the reply mentions "κόστος".
    assert (
        scrub_pricing(
            "Αυτο εδω",
            "από 40 τον καθένα σας να υπολογίζετε κόστος",
        )
        == "από [price] τον καθένα σας να υπολογίζετε κόστος"
    )


def test_scrub_pricing_replaces_bare_english_range_on_how_much():
    assert (
        scrub_pricing(
            "How much it would be this butterfly tattoo?",
            "Hello 110-130 depends the exact size",
        )
        == "Hello [price] depends the exact size"
    )


def test_scrub_pricing_replaces_bare_time_when_booking_signal_present():
    assert (
        scrub_pricing(
            "Θα μπορούσαμε να κλείσουμε ένα ραντεβού;",
            "Βεβαίως για άμεσα έχουμε τετάρτη 18:00",
        )
        == "Βεβαίως για άμεσα έχουμε τετάρτη [time]"
    )


def test_scrub_pricing_leaves_small_bare_numbers_alone_even_with_price_signal():
    # A single-digit number reads as a quantity, not a price, even on an
    # otherwise price-signalling line.
    assert (
        scrub_pricing("Πόσο κοστίζει;", "Θέλεις 2 σχέδια ή 1;")
        == "Θέλεις 2 σχέδια ή 1;"
    )


def test_scrub_pricing_leaves_bare_numbers_alone_without_any_signal():
    assert (
        scrub_pricing("Αυτο εδω θελω", "Ωραίο σχέδιο, στείλε και άλλες φωτο")
        == "Ωραίο σχέδιο, στείλε και άλλες φωτο"
    )


def test_scrub_pricing_matches_signal_word_regardless_of_accent_placement():
    # "κόστος" carries its stress accent on the letter right after "κ", so a
    # literal-accented stem would miss it even though "κοστίζει" (accent
    # further along) matches fine — accent-stripping must catch both.
    assert (
        scrub_pricing("θελω ενα τατου", "40 να υπολογίζετε κόστος")
        == "[price] να υπολογίζετε κόστος"
    )


def test_scrub_pricing_replaces_bare_time_next_to_a_day_name():
    # No "ραντεβού"/"ώρα" in either turn — the appointment context was
    # presumably set earlier in the thread, outside this two-turn window.
    assert (
        scrub_pricing(
            "Παρασκευή έχουμε βασικά τίποτα ;;;",
            "Παρασκευή 13:00 μπορείτε ;",
        )
        == "Παρασκευή [time] μπορείτε ;"
    )


def test_scrub_pricing_replaces_day_and_month_date():
    assert (
        scrub_pricing("γεια", "Έχουμε διαθέσιμο στις 19 Νοεμβρίου")
        == "Έχουμε διαθέσιμο στις [time]"
    )


def test_scrub_pricing_replaces_day_of_month_phrase():
    assert (
        scrub_pricing("γεια", "24 του μηνός μπορείτε")
        == "[time] μπορείτε"
    )


def test_scrub_pricing_replaces_date_unaccented():
    # Casual typing often drops accents entirely; the month-name match must
    # not depend on the accent being present.
    assert (
        scrub_pricing("γεια", "στις 19 Νοεμβριου μπορουμε")
        == "στις [time] μπορουμε"
    )


def test_scrub_pricing_leaves_weekday_names_alone():
    # Timeless vocabulary, not a stale fact — never scrubbed, even though
    # it's also used as a booking-signal word to gate bare-time scrubbing.
    assert (
        scrub_pricing("γεια", "Ραντεβού έχουμε Παρασκευή")
        == "Ραντεβού έχουμε Παρασκευή"
    )


def test_scrub_pricing_does_not_fragment_a_bare_time_when_price_signal_also_present():
    # Both a price signal ("€") and a booking signal ("ραντεβού") appear in
    # the same exchange — the bare-number pass must not run on "12:00"
    # before the bare-time pass treats it as one token.
    assert (
        scrub_pricing(
            "στα 75€ μπορούμε την Πέμπτη να βάλουμε ραντεβού;",
            "Καλησπέρα πέμπτη 12:00 μπορείτε ;",
        )
        == "Καλησπέρα πέμπτη [time] μπορείτε ;"
    )


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


def test_extract_pairs_drops_pair_whose_studio_turn_is_only_the_auto_reply():
    messages = [
        {"sender_name": "Maria", "content": "Γεια σας", "timestamp_ms": 1},
        {
            "sender_name": "2310tattoo studio by Christina",
            "content": AUTO_REPLY_TEXT,
            "timestamp_ms": 2,
        },
    ]
    assert extract_pairs("THREAD1", messages) == []


def test_extract_pairs_keeps_pair_when_real_content_follows_the_auto_reply():
    messages = [
        {"sender_name": "Maria", "content": "Γεια σας", "timestamp_ms": 1},
        {
            "sender_name": "2310tattoo studio by Christina",
            "content": AUTO_REPLY_TEXT,
            "timestamp_ms": 2,
        },
        {
            "sender_name": "2310tattoo studio by Christina",
            "content": "Καλησπέρα ναι βεβαίως στείλτε μας το σχέδιο",
            "timestamp_ms": 3,
        },
    ]
    pairs = extract_pairs("THREAD1", messages)
    assert len(pairs) == 1
    assert pairs[0].studio_reply_scrubbed == (
        AUTO_REPLY_TEXT + "\nΚαλησπέρα ναι βεβαίως στείλτε μας το σχέδιο"
    )


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
