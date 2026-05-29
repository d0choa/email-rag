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


def test_add_embedding_is_idempotent(db):
    # Re-adding the same chunk_id (resume/re-embed) must not raise and must not
    # duplicate the vector row.
    store = VectorStore(db)
    _insert_message(db, "<m1@x>")
    db.conn.execute(
        "INSERT INTO chunks(chunk_id, message_id, ord, text) VALUES (1, '<m1@x>', 0, 't')"
    )
    db.conn.commit()
    store.add_embedding(1, [1.0, 0.0, 0.0, 0.0])
    store.add_embedding(1, [0.0, 1.0, 0.0, 0.0])  # must not raise UNIQUE
    n = db.conn.execute("SELECT count(*) c FROM vec_chunks").fetchone()["c"]
    assert n == 1


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
    specs = [(1, "<a@x>", "alpha", [1.0, 0.0, 0.0, 0.0]),
             (2, "<b@x>", "beta", [0.9, 0.1, 0.0, 0.0]),
             (3, "<c@x>", "gamma", [0.0, 0.0, 1.0, 0.0])]
    for cid, mid, text, vec in specs:
        _insert_message(db, mid)
        db.conn.execute(
            "INSERT INTO chunks(chunk_id, message_id, ord, text) VALUES (?, ?, 0, ?)",
            (cid, mid, text),
        )
        db.conn.commit()
        store.add_embedding(cid, vec)
        store.add_fts(cid, text)

    results = store.hybrid_search(
        query_vec=[1.0, 0.0, 0.0, 0.0], query_text="gamma", k=3
    )
    ids = [r.chunk_id for r in results]
    assert set(ids) == {1, 2, 3}
    assert isinstance(results[0].score, float)


def test_hybrid_dedupes_to_one_chunk_per_message(db):
    # Two messages, two chunks each. Hybrid must return at most one per message.
    store = VectorStore(db)
    for mid in ("<m1@x>", "<m2@x>"):
        _insert_message(db, mid)
    rows = [(1, "<m1@x>", "alpha one"), (2, "<m1@x>", "alpha two"),
            (3, "<m2@x>", "alpha three"), (4, "<m2@x>", "alpha four")]
    for cid, mid, text in rows:
        db.conn.execute(
            "INSERT INTO chunks(chunk_id, message_id, ord, text) VALUES (?, ?, ?, ?)",
            (cid, mid, cid, text),
        )
        db.conn.commit()
        store.add_embedding(cid, [1.0, 0.0, 0.0, 0.0])
        store.add_fts(cid, text)

    results = store.hybrid_search([1.0, 0.0, 0.0, 0.0], "alpha", k=10)
    mids = [r.message_id for r in results]
    assert len(mids) == len(set(mids))  # no duplicate messages
    assert set(mids) == {"<m1@x>", "<m2@x>"}


def test_filtered_vector_search_ranks_within_filter(db):
    # The target sender's message is NOT a global nearest neighbour, yet a
    # --from filter must still surface it (regression for post-filter recall).
    store = VectorStore(db)
    for i in range(25):  # noise close to the query vector
        mid = f"<noise{i}@x>"
        _insert_message(db, mid, from_addr="other@x")
        db.conn.execute(
            "INSERT INTO chunks(chunk_id, message_id, ord, text) VALUES (?, ?, 0, 'noise')",
            (100 + i, mid),
        )
        db.conn.commit()
        store.add_embedding(100 + i, [1.0, 0.0, 0.0, 0.0])
    _insert_message(db, "<target@x>", from_addr="wanted@x")  # far from query
    db.conn.execute(
        "INSERT INTO chunks(chunk_id, message_id, ord, text) VALUES (1, '<target@x>', 0, 'target')"
    )
    db.conn.commit()
    store.add_embedding(1, [0.0, 1.0, 0.0, 0.0])

    hits = store.search_vectors([1.0, 0.0, 0.0, 0.0], k=5,
                                filters=Filters(from_addr="wanted@x"))
    assert [cid for cid, _ in hits] == [1]
