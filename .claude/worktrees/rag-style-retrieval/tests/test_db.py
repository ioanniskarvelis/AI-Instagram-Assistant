import sqlite3
import time

import pytest


def test_init_schema_is_idempotent():
    from app import db

    db.init_schema()
    db.init_schema()

    with db.connect() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
    assert row is not None


def test_init_schema_enables_wal():
    from app import db

    db.init_schema()

    with db.connect() as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_sweep_deletes_only_expired_rows():
    from app import db

    db.init_schema()
    now = int(time.time())
    with db.connect() as conn:
        conn.executemany(
            "INSERT INTO messages (sender_id, role, text, created_at)"
            " VALUES (?, ?, ?, ?)",
            [
                ("A", "user", "old", now - 21 * 86400),
                ("A", "user", "fresh", now - 19 * 86400),
            ],
        )

    deleted = db.sweep_expired(now=now, retention_days=20)

    assert deleted == 1
    with db.connect() as conn:
        remaining = [r[0] for r in conn.execute("SELECT text FROM messages")]
    assert remaining == ["fresh"]


def test_role_check_constraint_rejects_unknown_role():
    from app import db

    db.init_schema()
    with db.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO messages (sender_id, role, text, created_at)"
                " VALUES ('A', 'system', 'x', 0)"
            )
