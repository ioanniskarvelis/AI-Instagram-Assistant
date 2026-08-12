from app.trace import HistoryTurn, RetrievalHit, TraceRecord, get_trace, list_traces, save


def _record(**overrides) -> TraceRecord:
    defaults = dict(
        sender_id="SENDER_1",
        incoming_text="Πόσο κοστίζει;",
        history_window=[HistoryTurn(role="user", text="Πόσο κοστίζει;")],
        intent="price",
        intent_latency_ms=12.3,
        retrieval_hits=[RetrievalHit(question="q", reply="r", score=0.987)],
        retrieval_latency_ms=4.5,
        system_prompt="SYSTEM",
        reply="Η τέχνη θα σου απαντήσει.",
        reply_source="generated",
        llm_latency_ms=200.1,
        total_latency_ms=250.0,
    )
    defaults.update(overrides)
    return TraceRecord(**defaults)


async def test_save_and_get_trace_round_trips_all_fields():
    from app import db

    db.init_schema()
    assert await save(_record()) is True

    summaries = list_traces(limit=10, offset=0)
    assert len(summaries) == 1
    trace_id = summaries[0].id

    detail = get_trace(trace_id)
    assert detail is not None
    assert detail.sender_id == "SENDER_1"
    assert detail.incoming_text == "Πόσο κοστίζει;"
    assert detail.intent == "price"
    assert detail.intent_latency_ms == 12.3
    assert detail.retrieval_hits == [RetrievalHit(question="q", reply="r", score=0.987)]
    assert detail.retrieval_latency_ms == 4.5
    assert detail.system_prompt == "SYSTEM"
    assert detail.reply == "Η τέχνη θα σου απαντήσει."
    assert detail.reply_source == "generated"
    assert detail.llm_latency_ms == 200.1
    assert detail.total_latency_ms == 250.0
    assert detail.history_window == [HistoryTurn(role="user", text="Πόσο κοστίζει;")]


async def test_get_trace_returns_none_for_unknown_id():
    from app import db

    db.init_schema()
    assert get_trace(999) is None


async def test_list_traces_filters_by_sender_and_intent():
    from app import db

    db.init_schema()
    await save(_record(sender_id="A", intent="price"))
    await save(_record(sender_id="A", intent="booking"))
    await save(_record(sender_id="B", intent="price"))

    assert {t.sender_id for t in list_traces(limit=10, offset=0, sender_id="A")} == {"A"}
    assert len(list_traces(limit=10, offset=0, sender_id="A")) == 2
    assert len(list_traces(limit=10, offset=0, intent="price")) == 2
    assert len(list_traces(limit=10, offset=0, sender_id="A", intent="booking")) == 1


async def test_list_traces_orders_newest_first():
    from app import db

    db.init_schema()
    await save(_record(incoming_text="first"))
    await save(_record(incoming_text="second"))

    rows = list_traces(limit=10, offset=0)
    assert [r.incoming_text for r in rows] == ["second", "first"]


async def test_save_returns_false_and_never_raises_on_failure(monkeypatch):
    import sqlite3

    from app import trace

    def boom(_record):
        raise sqlite3.OperationalError("no such table")

    monkeypatch.setattr(trace, "_insert", boom)
    assert await save(_record()) is False
