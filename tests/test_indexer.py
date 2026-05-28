from datetime import datetime, timezone
from email_rag.indexer import Indexer
from email_rag.models import ParsedMessage
from email_rag.store import VectorStore


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.0, 0.0] for t in texts]


def _msg(mid, body, in_reply_to=None, references=None):
    return ParsedMessage(
        message_id=mid, from_addr="a@x", to_addrs=["b@y"], cc_addrs=[],
        subject="S", date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        body=body, in_reply_to=in_reply_to, references=references or [],
        folder="INBOX", uid=1,
    )


def test_store_messages_creates_chunks_and_threads(db):
    idx = Indexer(db, FakeEmbedder(), VectorStore(db))
    idx.store_messages([
        _msg("<root@x>", "first body"),
        _msg("<reply@x>", "second body", in_reply_to="<root@x>",
             references=["<root@x>"]),
    ])
    rows = db.conn.execute("SELECT thread_id FROM messages ORDER BY message_id").fetchall()
    assert {r["thread_id"] for r in rows} == {"<root@x>"}
    n_chunks = db.conn.execute("SELECT count(*) c FROM chunks").fetchone()["c"]
    assert n_chunks == 2
    pending = db.conn.execute("SELECT count(*) c FROM chunks WHERE embedded=0").fetchone()["c"]
    assert pending == 2


def test_embed_pending_marks_embedded(db):
    idx = Indexer(db, FakeEmbedder(), VectorStore(db))
    idx.store_messages([_msg("<m@x>", "hello world")])
    count = idx.embed_pending(batch_size=10)
    assert count == 1
    pending = db.conn.execute("SELECT count(*) c FROM chunks WHERE embedded=0").fetchone()["c"]
    assert pending == 0
    n_vec = db.conn.execute("SELECT count(*) c FROM vec_chunks").fetchone()["c"]
    assert n_vec == 1


def test_store_messages_is_idempotent(db):
    idx = Indexer(db, FakeEmbedder(), VectorStore(db))
    idx.store_messages([_msg("<m@x>", "hello world")])
    idx.store_messages([_msg("<m@x>", "hello world")])  # re-ingest same id
    n = db.conn.execute("SELECT count(*) c FROM messages").fetchone()["c"]
    assert n == 1
    n_chunks = db.conn.execute("SELECT count(*) c FROM chunks").fetchone()["c"]
    assert n_chunks == 1
