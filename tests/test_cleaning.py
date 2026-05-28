from email_rag.cleaning import clean_body


def test_strips_quoted_reply():
    text = (
        "Thanks, that works for me.\n"
        "\n"
        "On Mon, Jan 6, 2025 at 10:00 AM Alice <alice@x.com> wrote:\n"
        "> Can we meet Tuesday?\n"
        "> Let me know.\n"
    )
    cleaned = clean_body(text)
    assert "that works for me" in cleaned
    assert "Can we meet Tuesday" not in cleaned


def test_strips_signature():
    text = "See attached.\n\n-- \nAlice Smith\nDirector, Example Corp\n"
    cleaned = clean_body(text)
    assert "See attached." in cleaned
    assert "Director, Example Corp" not in cleaned


def test_plain_text_unchanged():
    text = "Just a simple note with no quotes."
    assert clean_body(text).strip() == text
