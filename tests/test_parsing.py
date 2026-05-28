from email_rag.parsing import parse_message

PLAIN = b"""From: Alice <alice@example.com>
To: Bob <bob@example.com>
Cc: carol@example.com
Subject: Project update
Date: Mon, 06 Jan 2025 10:00:00 +0000
Message-ID: <msg1@example.com>
In-Reply-To: <msg0@example.com>
References: <root@example.com> <msg0@example.com>
Content-Type: text/plain; charset=utf-8

Here is the update.
"""

HTML = b"""From: Alice <alice@example.com>
To: bob@example.com
Subject: HTML mail
Date: Mon, 06 Jan 2025 10:00:00 +0000
Message-ID: <msg2@example.com>
Content-Type: text/html; charset=utf-8

<html><body><p>Hello <b>world</b></p></body></html>
"""


def test_parse_plain():
    m = parse_message(PLAIN, folder="INBOX", uid=42)
    assert m.message_id == "<msg1@example.com>"
    assert m.from_addr == "alice@example.com"
    assert m.to_addrs == ["bob@example.com"]
    assert m.cc_addrs == ["carol@example.com"]
    assert m.subject == "Project update"
    assert m.in_reply_to == "<msg0@example.com>"
    assert m.references == ["<root@example.com>", "<msg0@example.com>"]
    assert "Here is the update." in m.body
    assert m.folder == "INBOX"
    assert m.uid == 42
    assert m.date.year == 2025


def test_parse_html_to_text():
    m = parse_message(HTML, folder="INBOX", uid=43)
    assert "Hello world" in m.body
    assert "<b>" not in m.body
