from email_rag.db import Database
from email_rag.models import Retrieved
from email_rag.store import _parse_date

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided email excerpts. "
    "Cite the excerpts you use with their bracketed numbers, e.g. [1], [2]. "
    "If the excerpts do not contain the answer, say "
    "'I couldn't find that in your email.' Never invent facts."
)


def build_context(retrieved: list[Retrieved], max_chars: int = 24000):
    """Format retrieved excerpts into a numbered context block + source list."""
    context_parts: list[str] = []
    sources: list[dict] = []
    used = 0
    for i, r in enumerate(retrieved, start=1):
        header = (
            f"[{i}] From {r.from_addr} · {r.date.date().isoformat()} · {r.subject}"
        )
        block = f"{header}\n{r.text}\n"
        if used + len(block) > max_chars and context_parts:
            break
        context_parts.append(block)
        used += len(block)
        sources.append(
            {
                "n": i,
                "message_id": r.message_id,
                "from": r.from_addr,
                "date": r.date.date().isoformat(),
                "subject": r.subject,
            }
        )
    return "\n".join(context_parts), sources


class Answerer:
    def __init__(self, db: Database, store, embedder, anthropic_client, model: str):
        self.db = db
        self.store = store
        self.embedder = embedder
        self.client = anthropic_client
        self.model = model

    def _expand_threads(self, retrieved: list[Retrieved], per_thread: int = 3):
        """Append sibling messages from each hit's thread for context; dedupe."""
        seen = {r.message_id for r in retrieved}
        expanded = list(retrieved)
        for r in list(retrieved):
            rows = self.db.conn.execute(
                "SELECT message_id, from_addr, subject, date, body, thread_id "
                "FROM messages WHERE thread_id=? AND message_id != ? "
                "ORDER BY date LIMIT ?",
                (r.thread_id, r.message_id, per_thread),
            ).fetchall()
            for row in rows:
                if row["message_id"] in seen:
                    continue
                seen.add(row["message_id"])
                expanded.append(
                    Retrieved(
                        chunk_id=-1,
                        message_id=row["message_id"],
                        text=row["body"] or "",
                        score=r.score * 0.5,
                        from_addr=row["from_addr"] or "",
                        subject=row["subject"] or "",
                        date=_parse_date(row["date"]),
                        thread_id=row["thread_id"] or row["message_id"],
                    )
                )
        expanded.sort(key=lambda x: x.score, reverse=True)
        return expanded

    def ask(self, question: str, k: int = 20, filters=None, max_chars: int = 24000):
        query_vec = self.embedder.embed_query(question)
        hits = self.store.hybrid_search(query_vec, question, k=k, filters=filters)
        if not hits:
            return "I couldn't find that in your email.", []
        expanded = self._expand_threads(hits)
        context, sources = build_context(expanded, max_chars=max_chars)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Email excerpts:\n\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        answer = "".join(
            block.text for block in message.content if hasattr(block, "text")
        )
        return answer, sources
