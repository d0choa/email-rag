from email_rag.models import Chunk, ParsedMessage

# ~500 tokens at ~4 chars/token.
DEFAULT_MAX_CHARS = 2000


def chunk_message(msg: ParsedMessage, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    body = (msg.body or "").strip()
    if not body:
        return []
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [body]

    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            pieces.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)

    return [Chunk(message_id=msg.message_id, ord=i, text=t) for i, t in enumerate(pieces)]
