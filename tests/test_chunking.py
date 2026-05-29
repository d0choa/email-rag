from datetime import datetime, timezone
from email_rag.models import ParsedMessage
from email_rag.chunking import chunk_message


def _msg(body: str) -> ParsedMessage:
    return ParsedMessage(
        message_id="<a@x>", from_addr="a@x", to_addrs=[], cc_addrs=[],
        subject="S", date=datetime(2025, 1, 1, tzinfo=timezone.utc), body=body,
    )


def test_short_message_is_one_chunk():
    chunks = chunk_message(_msg("a short body"), max_chars=2000)
    assert len(chunks) == 1
    assert chunks[0].ord == 0
    assert chunks[0].message_id == "<a@x>"
    assert chunks[0].text == "a short body"


def test_empty_body_yields_no_chunks():
    assert chunk_message(_msg("   ")) == []


def test_long_body_splits_on_paragraphs():
    para = "word " * 100  # ~500 chars
    body = "\n\n".join([para, para, para])
    chunks = chunk_message(_msg(body), max_chars=600)
    assert len(chunks) >= 3
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_single_oversized_paragraph_is_hard_split():
    # One unbroken block with no blank lines must still be capped per chunk.
    body = "word " * 2000  # ~10000 chars, single paragraph
    chunks = chunk_message(_msg(body), max_chars=1000)
    assert len(chunks) >= 10
    assert all(len(c.text) <= 1000 for c in chunks)
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_unbreakable_blob_is_hard_cut():
    # No spaces or newlines at all (e.g. a base64 blob) -> hard cut at max_chars.
    body = "x" * 5000
    chunks = chunk_message(_msg(body), max_chars=1000)
    assert all(len(c.text) <= 1000 for c in chunks)
    assert sum(len(c.text) for c in chunks) == 5000
