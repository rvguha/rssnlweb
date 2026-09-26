"""SQLite: items (append-only), per-source fetch validators, per-item embeddings.

This is the only state the server keeps. None of it is about users.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .models import Item

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_title TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    published_at TEXT,
    published_raw TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    collection TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS items_ingested ON items (ingested_at);
CREATE TABLE IF NOT EXISTS sources (
    name TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    etag TEXT,
    last_modified TEXT,
    sha256 TEXT,
    last_fetch TEXT,
    last_error TEXT
);
CREATE TABLE IF NOT EXISTS embeddings (
    item_id TEXT NOT NULL,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (item_id, model)
);
"""


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.executescript(_SCHEMA)
        # Databases created before the collection column existed.
        cols = {row[1] for row in self.db.execute("PRAGMA table_info(items)")}
        if "collection" not in cols:
            self.db.execute("ALTER TABLE items ADD COLUMN collection TEXT NOT NULL DEFAULT ''")

    async def setup(self) -> None:
        return None

    async def close(self) -> None:
        self.db.close()

    # --- items ---------------------------------------------------------------

    async def insert_missing(self, items: list[Item]) -> int:
        """Insert items whose id is new. Existing rows are never touched."""
        rows = [
            (
                item.id,
                item.source,
                item.source_title,
                item.url,
                item.title,
                item.content,
                _iso(item.published_at),
                item.published_raw,
                _iso(item.ingested_at),
                item.collection,
            )
            for item in items
        ]
        with self.db:
            cursor = self.db.executemany(
                "INSERT OR IGNORE INTO items VALUES (?,?,?,?,?,?,?,?,?,?)", rows
            )
        return cursor.rowcount

    def all_items(self) -> list[Item]:
        rows = self.db.execute("SELECT * FROM items ORDER BY ingested_at, id").fetchall()
        return [_row_item(row) for row in rows]

    async def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    def counts_by_source(self) -> dict[str, int]:
        rows = self.db.execute("SELECT source, COUNT(*) FROM items GROUP BY source").fetchall()
        return dict(rows)

    # --- sources -------------------------------------------------------------

    async def source_state(self, name: str) -> dict[str, str | None]:
        row = self.db.execute(
            "SELECT url, etag, last_modified, sha256, last_fetch, last_error "
            "FROM sources WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return {}
        keys = ("url", "etag", "last_modified", "sha256", "last_fetch", "last_error")
        return dict(zip(keys, row, strict=True))

    async def remember_source(
        self,
        name: str,
        url: str,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        sha256: str | None = None,
        error: str | None = None,
    ) -> None:
        previous = await self.source_state(name)
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?)",
                (
                    name,
                    url,
                    etag if error is None else previous.get("etag"),
                    last_modified if error is None else previous.get("last_modified"),
                    sha256 if error is None else previous.get("sha256"),
                    _iso(datetime.now(UTC)),
                    error,
                ),
            )

    async def all_sources(self) -> list[dict[str, str | None]]:
        rows = self.db.execute(
            "SELECT name, url, last_fetch, last_error FROM sources ORDER BY name"
        ).fetchall()
        return [
            {"name": r[0], "url": r[1], "last_fetch": r[2], "last_error": r[3]} for r in rows
        ]

    # --- embeddings ----------------------------------------------------------

    def embeddings(self, model: str, ids: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            rows = self.db.execute(
                f"SELECT item_id, vector FROM embeddings WHERE model = ? AND item_id IN ({marks})",
                (model, *chunk),
            ).fetchall()
            for item_id, blob in rows:
                found[item_id] = np.frombuffer(blob, dtype=np.float32)
        return found

    def save_embeddings(self, model: str, vectors: dict[str, np.ndarray]) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?,?,?)",
                [
                    (item_id, model, np.asarray(v, dtype=np.float32).tobytes())
                    for item_id, v in vectors.items()
                ],
            )


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value is not None else None


def _row_item(row: tuple) -> Item:
    return Item(
        id=row[0],
        source=row[1],
        source_title=row[2],
        url=row[3],
        title=row[4],
        content=row[5],
        published_at=datetime.fromisoformat(row[6]) if row[6] else None,
        published_raw=row[7],
        ingested_at=datetime.fromisoformat(row[8]),
        collection=row[9] if len(row) > 9 else "",
    )
