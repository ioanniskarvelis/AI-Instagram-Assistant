async def test_image_urls_for_returns_stored_urls_oldest_first():
    from app import db
    from app.quotes import image_urls_for, record_attachment

    db.init_schema()
    assert await record_attachment("SENDER_1", "https://cdn/a.jpg") is True
    assert await record_attachment("SENDER_1", "https://cdn/b.jpg") is True

    assert await image_urls_for("SENDER_1") == [
        "https://cdn/a.jpg",
        "https://cdn/b.jpg",
    ]


async def test_image_urls_for_respects_limit_keeping_most_recent():
    from app import db
    from app.quotes import image_urls_for, record_attachment

    db.init_schema()
    for i in range(3):
        await record_attachment("SENDER_1", f"https://cdn/{i}.jpg")

    assert await image_urls_for("SENDER_1", limit=2) == [
        "https://cdn/1.jpg",
        "https://cdn/2.jpg",
    ]


async def test_image_urls_for_unknown_sender_is_empty():
    from app import db
    from app.quotes import image_urls_for

    db.init_schema()
    assert await image_urls_for("NOBODY") == []


async def test_senders_do_not_see_each_others_attachments():
    from app import db
    from app.quotes import image_urls_for, record_attachment

    db.init_schema()
    await record_attachment("SENDER_1", "https://cdn/mine.jpg")
    await record_attachment("SENDER_2", "https://cdn/theirs.jpg")

    assert await image_urls_for("SENDER_1") == ["https://cdn/mine.jpg"]


async def test_resolve_quote_request_returns_sender_and_marks_answered():
    from app import db
    from app.quotes import create_quote_request, resolve_quote_request

    db.init_schema()
    assert await create_quote_request("SENDER_1", 4242) is True

    assert await resolve_quote_request(4242) == "SENDER_1"
    # Answered once; a duplicate Telegram delivery of the same reply must
    # not resolve (and re-announce) it a second time.
    assert await resolve_quote_request(4242) is None


async def test_resolve_quote_request_unknown_message_id_is_none():
    from app import db
    from app.quotes import resolve_quote_request

    db.init_schema()
    assert await resolve_quote_request(999) is None


async def test_record_attachment_failure_is_swallowed(monkeypatch):
    from app import db, quotes

    db.init_schema()

    def boom():
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(quotes.db, "connect", boom)

    assert await quotes.record_attachment("SENDER_1", "https://cdn/a.jpg") is False
    assert await quotes.image_urls_for("SENDER_1") == []
    assert await quotes.create_quote_request("SENDER_1", 1) is False
    assert await quotes.resolve_quote_request(1) is None
