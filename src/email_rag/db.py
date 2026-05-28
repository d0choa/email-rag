import sqlite3

import sqlite_vec

SCHEMA_VERSION = 1


class Database:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

    def init_schema(self, dim: int = 768) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS messages(
                message_id   TEXT PRIMARY KEY,
                thread_id    TEXT,
                from_addr    TEXT,
                to_addrs     TEXT,
                cc_addrs     TEXT,
                subject      TEXT,
                date         TEXT,
                folder       TEXT,
                uid          INTEGER,
                body         TEXT,
                attachments  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
            CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date);

            CREATE TABLE IF NOT EXISTS chunks(
                chunk_id   INTEGER PRIMARY KEY,
                message_id TEXT NOT NULL REFERENCES messages(message_id),
                ord        INTEGER NOT NULL,
                text       TEXT NOT NULL,
                embedded   INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_msg ON chunks(message_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_embedded ON chunks(embedded);

            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding float[{dim}]
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text);

            CREATE TABLE IF NOT EXISTS sync_state(
                folder        TEXT PRIMARY KEY,
                uidvalidity   INTEGER,
                last_seen_uid INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS meta(
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        cur.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value)
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
