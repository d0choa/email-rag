import re
import sqlite3
from datetime import datetime

import numpy as np
import sqlite_vec

from email_rag.db import Database
from email_rag.models import Retrieved

RRF_K = 60


def _fts_query(text: str) -> str:
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    # OR-join so a natural-language query matches on any term (recall-oriented);
    # space-joining would be AND, requiring every token present.
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)


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
            until_bound = (
                self.until if "T" in self.until else self.until + "T23:59:59.999999"
            )
            clauses.append("m.date <= ?")
            params.append(until_bound)
        if self.folder:
            clauses.append("m.folder = ?")
            params.append(self.folder)
        return (" AND ".join(clauses), params) if clauses else ("", [])


class VectorStore:
    def __init__(self, db: Database):
        self.db = db

    def add_embedding(self, chunk_id: int, vector: list[float]) -> None:
        # vec0 virtual tables ignore "INSERT OR REPLACE" conflict handling and
        # raise UNIQUE on a duplicate rowid, so delete-then-insert (as for FTS).
        self.db.conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (chunk_id,))
        self.db.conn.execute(
            "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(vector)),
        )

    def add_fts(self, chunk_id: int, text: str) -> None:
        self.db.conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
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
        if allowed is None:
            # Unfiltered: use the indexed vec0 KNN.
            rows = self.db.conn.execute(
                "SELECT rowid, distance FROM vec_chunks "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (sqlite_vec.serialize_float32(query_vec), k),
            ).fetchall()
            return [(r["rowid"], r["distance"]) for r in rows]
        if not allowed:
            return []
        # Filtered: rank exactly WITHIN the allowed set. vec0 KNN can't be
        # constrained to a metadata subset, and post-filtering a global top-k
        # silently drops matches that aren't global nearest neighbours.
        return self._exact_filtered_knn(query_vec, k, allowed)

    def _exact_filtered_knn(self, query_vec, k, allowed):
        ids = list(allowed)
        qv = np.asarray(query_vec, dtype=np.float32)
        cids: list[int] = []
        mats: list = []
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self.db.conn.execute(
                f"SELECT rowid, embedding FROM vec_chunks WHERE rowid IN ({placeholders})",
                batch,
            ).fetchall()
            for r in rows:
                cids.append(r["rowid"])
                mats.append(np.frombuffer(r["embedding"], dtype=np.float32))
        if not cids:
            return []
        dists = np.linalg.norm(np.vstack(mats) - qv, axis=1)
        order = np.argsort(dists)[:k]
        return [(int(cids[i]), float(dists[i])) for i in order]

    def search_fts(self, query_text, k, filters=None) -> list[int]:
        match = _fts_query(query_text)
        if not match:
            return []
        where, params = filters.where() if filters else ("", [])
        sql = "SELECT f.rowid FROM chunks_fts f"
        if where:
            # Push the metadata filter into SQL so FTS ranks within the subset
            # instead of post-filtering a global top-k.
            sql += (
                " JOIN chunks c ON c.chunk_id = f.rowid"
                " JOIN messages m ON m.message_id = c.message_id"
            )
        sql += " WHERE chunks_fts MATCH ?"
        args: list = [match]
        if where:
            sql += f" AND {where}"
            args += params
        sql += " ORDER BY rank LIMIT ?"
        args.append(k)
        try:
            rows = self.db.conn.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r["rowid"] for r in rows]

    def hybrid_search(self, query_vec, query_text, k, filters=None):
        # Over-retrieve per arm so that, after collapsing to one chunk per
        # message, we still have ~k distinct messages to return.
        pool = max(k * 3, 30)
        vec_hits = self.search_vectors(query_vec, pool, filters)
        fts_hits = self.search_fts(query_text, pool, filters)

        scores: dict[int, float] = {}
        for rank, (cid, _dist) in enumerate(vec_hits):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        for rank, cid in enumerate(fts_hits):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        # Collapse to the best-scoring chunk per message; keep top-k messages.
        out: list[Retrieved] = []
        seen: set[str] = set()
        for r in self._hydrate(ranked):
            if r.message_id in seen:
                continue
            seen.add(r.message_id)
            out.append(r)
            if len(out) >= k:
                break
        return out

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
