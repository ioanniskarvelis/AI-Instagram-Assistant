import pytest


async def test_append_and_read_round_trip():
    from app import db
    from app.history import Turn, append, recent

    db.init_schema()
    assert await append("SENDER_1", "user", "γεια") is True
    assert await append("SENDER_1", "assistant", "γεια σου") is True

    assert await recent("SENDER_1", 10) == [
        Turn(role="user", text="γεια"),
        Turn(role="assistant", text="γεια σου"),
    ]


async def test_recent_returns_only_the_last_n_turns():
    from app import db
    from app.history import append, recent

    db.init_schema()
    for index in range(5):
        await append("SENDER_1", "user", f"m{index}")

    turns = await recent("SENDER_1", 2)

    assert [turn.text for turn in turns] == ["m3", "m4"]


async def test_recent_drops_leading_assistant_turns():
    from app import db
    from app.history import append, recent

    db.init_schema()
    await append("SENDER_1", "user", "first")
    await append("SENDER_1", "assistant", "reply")
    await append("SENDER_1", "user", "second")

    # A window of 2 would start on the assistant turn, which the API rejects.
    turns = await recent("SENDER_1", 2)

    assert [turn.role for turn in turns] == ["user"]
    assert turns[0].text == "second"


async def test_recent_for_unknown_sender_is_empty():
    from app import db
    from app.history import recent

    db.init_schema()
    assert await recent("NOBODY", 10) == []


async def test_senders_do_not_see_each_other():
    from app import db
    from app.history import append, recent

    db.init_schema()
    await append("SENDER_1", "user", "mine")
    await append("SENDER_2", "user", "theirs")

    turns = await recent("SENDER_1", 10)

    assert [turn.text for turn in turns] == ["mine"]


async def test_store_failures_are_swallowed(monkeypatch):
    from app import db, history

    db.init_schema()

    def boom():
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(history.db, "connect", boom)

    assert await history.append("SENDER_1", "user", "x") is False
    assert await history.recent("SENDER_1", 10) == []
