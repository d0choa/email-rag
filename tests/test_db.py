import sqlite_vec
from email_rag.db import Database, SCHEMA_VERSION


def test_schema_creates_tables(db):
    names = {
        r["name"]
        for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }
    assert {"messages", "chunks", "vec_chunks", "chunks_fts", "sync_state", "meta"} <= names


def test_meta_records_schema_version(db):
    row = db.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert int(row["value"]) == SCHEMA_VERSION


def test_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "deeper" / "index.db"
    database = Database(str(path))
    database.init_schema(dim=4)
    assert path.exists()
    database.close()


def test_vec_table_accepts_vector(db):
    db.conn.execute(
        "INSERT INTO vec_chunks(rowid, embedding) VALUES (1, ?)",
        (sqlite_vec.serialize_float32([0.1, 0.2, 0.3, 0.4]),),
    )
    db.conn.commit()
    n = db.conn.execute("SELECT count(*) AS c FROM vec_chunks").fetchone()["c"]
    assert n == 1
