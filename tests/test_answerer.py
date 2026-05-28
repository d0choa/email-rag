from datetime import datetime, timezone
from email_rag.answerer import Answerer, build_context
from email_rag.models import Retrieved


def _ret(cid, mid, text, tid):
    return Retrieved(
        chunk_id=cid, message_id=mid, text=text, score=1.0 / cid,
        from_addr="alice@x.com", subject="Subj",
        date=datetime(2025, 1, 6, tzinfo=timezone.utc), thread_id=tid,
    )


def test_build_context_formats_citations():
    rets = [_ret(1, "<m1@x>", "the dataset ships in March", "<m1@x>")]
    context, sources = build_context(rets, max_chars=10000)
    assert "[1]" in context
    assert "alice@x.com" in context
    assert "the dataset ships in March" in context
    assert sources[0]["n"] == 1
    assert sources[0]["message_id"] == "<m1@x>"


class FakeAnthropic:
    def __init__(self):
        self.messages = self
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs

        class R:
            content = [type("B", (), {"text": "March, per [1]."})()]

        return R()


def test_ask_returns_answer_and_sources(db):
    # seed one message + chunk so thread expansion has something to read
    db.conn.execute(
        "INSERT INTO messages(message_id, thread_id, from_addr, to_addrs, cc_addrs,"
        " subject, date, folder, uid, body, attachments)"
        " VALUES ('<m1@x>','<m1@x>','alice@x.com','','','Subj',"
        "'2025-01-06T00:00:00+00:00','INBOX',1,'the dataset ships in March','')"
    )
    db.conn.commit()

    class FakeStore:
        def hybrid_search(self, query_vec, query_text, k, filters=None):
            return [_ret(1, "<m1@x>", "the dataset ships in March", "<m1@x>")]

    class FakeEmbedder:
        def embed_query(self, text):
            return [1.0, 0.0, 0.0, 0.0]

    ans = Answerer(db, FakeStore(), FakeEmbedder(), FakeAnthropic(), model="claude-sonnet-4-6")
    answer, sources = ans.ask("when does the dataset ship?", k=5)
    assert "March" in answer
    assert sources[0]["message_id"] == "<m1@x>"
