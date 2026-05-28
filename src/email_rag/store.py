from datetime import datetime

import sqlite_vec

from email_rag.db import Database
from email_rag.models import Retrieved

RRF_K = 60


def _parse_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return datetime(1970, 1, 1)


class Filters:
    def __init__(self, from_addr=None, since=None, until=None, folder=None):
        self.from_addr = from_addr
        self.since = since
        self.until = until
        self.folder = folder

    def where(self) -> tuple[str, list]:
        clauses, params = [], []
        if self.from_addr:
            clauses.append("m.from_addr LIKE ?")
            params.append(f"%{self.from_addr.lower()}%")
        if self.since:
            clauses.append("m.date >= ?")
            params.append(self.since)
        if self.until:
            clauses.append("m.date <= ?")
            params.append(self.until)
        if self.folder:
            clauses.append("m.folder = ?")
            params.append(self.folder)
        return (" AND ".join(clauses), params) if clauses else ("", [])


class VectorStore:
    def __init__(self, db: Database):
        self.db = db

    def add_embedding(self, chunk_id: int, vector: list[float]) -> None:
        self.db.conn.execute(
            "INSERT OR REPLACE INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(vector)),
        )

    def add_fts(self, chunk_id: int, text: str) -> None:
        self.db.conn.execute(
            "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", (chunk_id, text)
        )

    def _allowed_ids(self, filters: "Filters | None") -> "set[int] | None":
        if not filters:
            return None
        where, params = filters.where()
        if not where:
            return None
        rows = self.db.conn.execute(
            "SELECT c.chunk_id FROM chunks c "
            "JOIN messages m ON m.message_id = c.message_id "
            f"WHERE {where}",
            params,
        ).fetchall()
        return {r["chunk_id"] for r in rows}

    def search_vectors(self, query_vec, k, filters=None):
        allowed = self._allowed_ids(filters)
        oversample = k * 5 if allowed is not None else k
        rows = self.db.conn.execute(
            "SELECT rowid, distance FROM vec_chunks "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (sqlite_vec.serialize_float32(query_vec), oversample),
        ).fetchall()
        hits = [(r["rowid"], r["distance"]) for r in rows]
        if allowed is not None:
            hits = [h for h in hits if h[0] in allowed]
        return hits[:k]

    def search_fts(self, query_text, k, filters=None) -> list[int]:
        allowed = self._allowed_ids(filters)
        rows = self.db.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (query_text, k * 5 if allowed is not None else k),
        ).fetchall()
        ids = [r["rowid"] for r in rows]
        if allowed is not None:
            ids = [i for i in ids if i in allowed]
        return ids[:k]

    def hybrid_search(self, query_vec, query_text, k, filters=None):
        vec_hits = self.search_vectors(query_vec, k, filters)
        fts_hits = self.search_fts(query_text, k, filters)

        scores: dict[int, float] = {}
        for rank, (cid, _dist) in enumerate(vec_hits):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        for rank, cid in enumerate(fts_hits):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return self._hydrate(ranked)

    def _hydrate(self, ranked: list) -> list:
        out: list[Retrieved] = []
        for cid, score in ranked:
            row = self.db.conn.execute(
                "SELECT c.chunk_id, c.text, m.message_id, m.from_addr, m.subject,"
                " m.date, m.thread_id "
                "FROM chunks c JOIN messages m ON m.message_id = c.message_id "
                "WHERE c.chunk_id = ?",
                (cid,),
            ).fetchone()
            if not row:
                continue
            out.append(
                Retrieved(
                    chunk_id=row["chunk_id"],
                    message_id=row["message_id"],
                    text=row["text"],
                    score=score,
                    from_addr=row["from_addr"] or "",
                    subject=row["subject"] or "",
                    date=_parse_date(row["date"]),
                    thread_id=row["thread_id"] or row["message_id"],
                )
            )
        return out
