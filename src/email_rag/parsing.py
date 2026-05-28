from datetime import datetime, timezone
from email import message_from_bytes
from email.header import decode_header, make_header
from email.message import Message
from email.utils import getaddresses, parsedate_to_datetime

from selectolax.parser import HTMLParser

from email_rag.models import ParsedMessage


def _decode(value: str | None) -> str:
    """Decode an RFC 2047 header, tolerating mislabeled or invalid charsets.

    Real mailboxes contain headers whose declared charset is wrong (e.g. bytes
    that are illegal for the announced `gb2312`). stdlib `make_header` decodes
    strictly and raises; we decode each part ourselves with errors="replace"
    and fall back to utf-8 for unknown charsets so no single header aborts a sync.
    """
    if not value:
        return ""
    try:
        parts: list[str] = []
        for raw, charset in decode_header(value):
            if isinstance(raw, bytes):
                try:
                    parts.append(raw.decode(charset or "utf-8", errors="replace"))
                except LookupError:
                    parts.append(raw.decode("utf-8", errors="replace"))
            else:
                parts.append(raw)
        return "".join(parts)
    except Exception:
        return value if isinstance(value, str) else str(value)


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
