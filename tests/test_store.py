from datetime import datetime, timezone
from email_rag.store import Filters, VectorStore


def _insert_message(db, mid, body="b", date="2025-01-01T00:00:00+00:00",
                    from_addr="a@x", thread_id=None):
    db.conn.execute(
        "INSERT INTO messages(message_id, thread_id, from_addr, to_addrs, cc_addrs,"
        " subject, date, folder, uid, body, attachments)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (mid, thread_id or mid, from_addr, "", "", "Subj", date, "INBOX", 1, body, ""),
    )
    db.conn.commit()


def test_add_and_vector_search(db):
    store = VectorStore(db)
    _insert_message(db, "<m1@x>")
    db.conn.execute(
        "INSERT INTO chunks(chunk_id, message_id, ord, text) VALUES (1, '<m1@x>', 0, 'cat dog')"
    )
    db.conn.execute(
        "INSERT INTO chunks(chunk_id, message_id, ord, text) VALUES (2, '<m1@x>', 1, 'fish bird')"
    )
    db.conn.commit()
    store.add_embedding(1, [1.0, 0.0, 0.0, 0.0])
    store.add_embedding(2, [0.0, 1.0, 0.0, 0.0])
    store.add_fts(1, "cat dog")
    store.add_fts(2, "fish bird")

    hits = store.search_vectors([1.0, 0.0, 0.0, 0.0], k=2)
    assert hits[0][0] == 1  # nearest chunk_id first


def test_fts_search(db):
    store = VectorStore(db)
    _insert_message(db, "<m1@x>")
    db.conn.execute(
        "INSERT INTO chunks(chunk_id, message_id, ord, text) VALUES (1, '<m1@x>', 0, 'budget spreadsheet')"
    )
    db.conn.commit()
    store.add_fts(1, "budget spreadsheet")
    assert 1 in store.search_fts("budget", k=5)


def test_fts_natural_language_query_does_not_crash(db):
    store = VectorStore(db)
    _insert_message(db, "<m1@x>")
    db.conn.execute(
        "INSERT INTO chunks(chunk_id, message_id, ord, text) VALUES (1, '<m1@x>', 0, 'the quarterly budget spreadsheet')"
    )
    db.conn.commit()
    store.add_fts(1, "the quarterly budget spreadsheet")
    hits = store.search_fts("what's the budget?", k=5)
    assert 1 in hits
    assert store.search_fts("???", k=5) == []


def test_until_includes_end_day(db):
    store = VectorStore(db)
    _insert_message(db, "<jan10@x>", date="2025-01-10T14:30:00+00:00")
    _insert_message(db, "<jan11@x>", date="2025-01-11T09:00:00+00:00")
    db.conn.execute(
        "INSERT INTO chunks(chunk_id, message_id, ord, text) VALUES (1, '<jan10@x>', 0, 'tenth day')"
    )
    db.conn.execute(
        "INSERT INTO chunks(chunk_id, message_id, ord, text) VALUES (2, '<jan11@x>', 0, 'eleventh day')"
    )
    db.conn.commit()
    store.add_embedding(1, [1.0, 0.0, 0.0, 0.0])
    store.add_embedding(2, [0.0, 1.0, 0.0, 0.0])
    store.add_fts(1, "tenth day")
    store.add_fts(2, "eleventh day")

    filters = Filters(until="2025-01-10")
    results = store.hybrid_search(
        query_vec=[1.0, 0.0, 0.0, 0.0], query_text="day", k=5, filters=filters
    )
    mids = {r.message_id for r in results}
    assert "<jan10@x>" in mids
    assert "<jan11@x>" not in mids


def test_hybrid_merges_by_rrf(db):
    store = VectorStore(db)
    _insert_message(db, "<m1@x>")
    for cid, text in [(1, "alpha"), (2, "beta"), (3, "gamma")]:
        db.conn.execute(
            "INSERT INTO chunks(chunk_id, message_id, ord, text) VALUES (?, '<m1@x>', ?, ?)",
            (cid, cid, text),
        )
    db.conn.commit()
    store.add_embedding(1, [1.0, 0.0, 0.0, 0.0])
    store.add_embedding(2, [0.9, 0.1, 0.0, 0.0])
    store.add_embedding(3, [0.0, 0.0, 1.0, 0.0])
    for cid, text in [(1, "alpha"), (2, "beta"), (3, "gamma")]:
        store.add_fts(cid, text)

    results = store.hybrid_search(
        query_vec=[1.0, 0.0, 0.0, 0.0], query_text="gamma", k=3
    )
    ids = [r.chunk_id for r in results]
    assert set(ids) == {1, 2, 3}
    assert results[0].message_id == "<m1@x>"
    assert isinstance(results[0].score, float)
