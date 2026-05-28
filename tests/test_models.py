from datetime import datetime, timezone
from email_rag.models import ParsedMessage, Chunk, Retrieved


def test_parsed_message_defaults():
    m = ParsedMessage(
        message_id="<a@x>",
        from_addr="a@x.com",
        to_addrs=["b@y.com"],
        cc_addrs=[],
        subject="Hi",
        date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        body="hello",
    )
    assert m.references == []
    assert m.attachment_names == []
    assert m.thread_id is None
    assert m.uid == 0


def test_chunk_and_retrieved():
    c = Chunk(message_id="<a@x>", ord=0, text="t")
    assert c.chunk_id is None
    r = Retrieved(
        chunk_id=1, message_id="<a@x>", text="t", score=0.5,
        from_addr="a@x.com", subject="Hi",
        date=datetime(2025, 1, 1, tzinfo=timezone.utc), thread_id="<a@x>",
    )
    assert r.score == 0.5
