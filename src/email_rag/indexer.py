from email_rag.chunking import chunk_message
from email_rag.cleaning import clean_body
from email_rag.db import Database
from email_rag.models import ParsedMessage
from email_rag.store import VectorStore
from email_rag.threading_ import thread_id_for


class Indexer:
    def __init__(self, db: Database, embedder, store: VectorStore):
        self.db = db
        self.embedder = embedder
        self.store = store

    def store_messages(self, messages: list[ParsedMessage]) -> int:
        """Persist messages and their (unembedded) chunks. Idempotent by message_id."""
        stored = 0
        for msg in messages:
            if not msg.message_id:
                continue
            exists = self.db.conn.execute(
                "SELECT 1 FROM messages WHERE message_id=?", (msg.message_id,)
            ).fetchone()
            if exists:
                continue
            tid = thread_id_for(msg.message_id, msg.in_reply_to, msg.references)
            cleaned = clean_body(msg.body)
            self.db.conn.execute(
                "INSERT INTO messages(message_id, thread_id, from_addr, to_addrs,"
                " cc_addrs, subject, date, folder, uid, body, attachments)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    msg.message_id, tid, msg.from_addr, ",".join(msg.to_addrs),
                    ",".join(msg.cc_addrs), msg.subject, msg.date.isoformat(),
                    msg.folder, msg.uid, cleaned, ",".join(msg.attachment_names),
                ),
            )
            cleaned_msg = ParsedMessage(**{**msg.__dict__, "body": cleaned})
            for chunk in chunk_message(cleaned_msg):
                self.db.conn.execute(
                    "INSERT INTO chunks(message_id, ord, text, embedded)"
                    " VALUES (?,?,?,0)",
                    (chunk.message_id, chunk.ord, chunk.text),
                )
            stored += 1
        self.db.conn.commit()
        return stored

    def embed_pending(self, batch_size: int = 64) -> int:
        """Embed all chunks with embedded=0, writing vectors + FTS per batch."""
        total = 0
        while True:
            rows = self.db.conn.execute(
                "SELECT chunk_id, text FROM chunks WHERE embedded=0 LIMIT ?",
                (batch_size,),
            ).fetchall()
            if not rows:
                break
            ids = [r["chunk_id"] for r in rows]
            texts = [r["text"] for r in rows]
            try:
                vectors = self.embedder.embed_documents(texts)
                pairs = list(zip(ids, texts, vectors))
                failed_ids: list[int] = []
            except Exception:
                # A bad item fails the whole batch request; retry one at a time
                # so a single unembeddable chunk can't abort the run.
                pairs, failed_ids = self._embed_individually(ids, texts)
            for cid, text, vec in pairs:
                self.store.add_embedding(cid, vec)
                self.store.add_fts(cid, text)
                self.db.conn.execute(
                    "UPDATE chunks SET embedded=1 WHERE chunk_id=?", (cid,)
                )
            for cid in failed_ids:
                # Sentinel 2 = "tried, model rejected" — keeps it out of the
                # embedded=0 work queue so the run terminates.
                self.db.conn.execute(
                    "UPDATE chunks SET embedded=2 WHERE chunk_id=?", (cid,)
                )
            self.db.conn.commit()
            total += len(pairs)
        return total

    def _embed_individually(self, ids, texts):
        """Embed chunks one by one, isolating any the model rejects.

        Returns (succeeded_triples, failed_ids) where succeeded_triples are
        (chunk_id, text, vector) and failed_ids are chunk_ids the model rejected.
        """
        pairs, failed_ids = [], []
        for cid, text in zip(ids, texts):
            try:
                vec = self.embedder.embed_documents([text])[0]
            except Exception:
                failed_ids.append(cid)
                continue
            pairs.append((cid, text, vec))
        return pairs, failed_ids
