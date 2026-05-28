from datetime import datetime, timezone
from email import message_from_bytes
from email.header import decode_header, make_header
from email.message import Message
from email.utils import getaddresses, parsedate_to_datetime

from selectolax.parser import HTMLParser

from email_rag.models import ParsedMessage


def _decode(value: str | None) -> str:
    if not value:
        return ""
    return str(make_header(decode_header(value)))


def _addrs(msg: Message, header: str) -> list[str]:
    raw = msg.get_all(header, [])
    return [addr.lower() for _, addr in getaddresses(raw) if addr]


def _refs(value: str | None) -> list[str]:
    if not value:
        return []
    return value.split()


def _html_to_text(html: str) -> str:
    return HTMLParser(html).text(separator=" ", strip=True)


def _extract_body(msg: Message) -> tuple[str, list[str]]:
    attachments: list[str] = []
    plain_parts: list[str] = []
    html_parts: list[str] = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        disp = (part.get_content_disposition() or "").lower()
        filename = part.get_filename()
        if disp == "attachment" or filename:
            if filename:
                attachments.append(_decode(filename))
            continue
        ctype = part.get_content_type()
        try:
            payload = part.get_payload(decode=True)
            text = payload.decode(part.get_content_charset() or "utf-8", "replace")
        except Exception:
            continue
        if ctype == "text/plain":
            plain_parts.append(text)
        elif ctype == "text/html":
            html_parts.append(text)
    if plain_parts:
        return "\n".join(plain_parts).strip(), attachments
    if html_parts:
        return _html_to_text("\n".join(html_parts)).strip(), attachments
    return "", attachments


def parse_message(raw: bytes, folder: str = "", uid: int = 0) -> ParsedMessage:
    msg = message_from_bytes(raw)
    body, attachments = _extract_body(msg)
    try:
        date = parsedate_to_datetime(msg.get("Date"))
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        date = datetime(1970, 1, 1, tzinfo=timezone.utc)
    from_addrs = _addrs(msg, "From")
    return ParsedMessage(
        message_id=(msg.get("Message-ID") or "").strip(),
        from_addr=from_addrs[0] if from_addrs else "",
        to_addrs=_addrs(msg, "To"),
        cc_addrs=_addrs(msg, "Cc"),
        subject=_decode(msg.get("Subject")),
        date=date,
        body=body,
        in_reply_to=(msg.get("In-Reply-To") or "").strip() or None,
        references=_refs(msg.get("References")),
        attachment_names=attachments,
        folder=folder,
        uid=uid,
    )
