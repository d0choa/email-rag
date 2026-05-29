from email_rag.models import Chunk, ParsedMessage

# ~500 tokens at ~4 chars/token.
DEFAULT_MAX_CHARS = 2000


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Split a single over-long unit into <=max_chars pieces.

    Prefer to break at the last newline or space before the limit so we don't
    cut mid-word; fall back to a hard cut when there is no boundary (e.g. a long
    unbroken base64 blob).
    """
    pieces: list[str] = []
    while len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        head = text[:cut].strip()
        if head:
            pieces.append(head)
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces


def chunk_message(msg: ParsedMessage, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    body = (msg.body or "").strip()
    if not body:
        return []
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [body]

    # Pre-split any single paragraph that alone exceeds the limit, so every
    # unit fed to the packer is <=max_chars and thus every emitted chunk is too.
    units: list[str] = []
    for para in paragraphs:
        if len(para) > max_chars:
            units.extend(_split_oversized(para, max_chars))
        else:
            units.append(para)

    pieces: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 2 > max_chars:
            pieces.append(current)
            current = unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        pieces.append(current)

    return [Chunk(message_id=msg.message_id, ord=i, text=t) for i, t in enumerate(pieces)]
