"""Serve chosen collections from process memory instead of a Cosmos vector query.

A collection that is small, grows slowly and is read a lot (This American Life: 28k passages, a new episode a
week) gains nothing from a database round trip on every query and is exposed to its throttling and timeouts. Its
passages and vectors are loaded once at startup from a snapshot (scripts/export_snapshot.py) and searched with the
same in-memory Index the SQLite mode uses. Whatever was ingested after the snapshot is fetched from Cosmos at
startup and again every few minutes, so the snapshot only has to be rebuilt to keep that catch-up small.

Other collections, and any query without a collection, still go to Cosmos. If a snapshot cannot be loaded the
collection falls back to Cosmos and says so in the log and in /health, rather than failing the query.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np

from .index import Index
from .models import Candidate, Item

logger = logging.getLogger("qdrss")
METADATA_TOKEN = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"


def write_snapshot(directory: Path, collection: str, items: list[Item], matrix: np.ndarray) -> None:
    """items.jsonl.gz + float16 vectors (half the size; cosine ranking does not notice)."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = (json.dumps({**asdict(i), "published_at": i.published_at and i.published_at.isoformat(),
                        "ingested_at": i.ingested_at.isoformat()}) for i in items)
    with gzip.open(directory / f"{collection}.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write("\n".join(rows))
    np.save(directory / f"{collection}.npy", matrix.astype(np.float16))


def parse_snapshot(items_gz: bytes, vectors_npy: bytes) -> tuple[list[Item], np.ndarray]:
    lines = gzip.decompress(items_gz).decode("utf-8").splitlines()
    items = [Item(**{**row, "published_at": datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
                     "ingested_at": datetime.fromisoformat(row["ingested_at"])})
             for row in map(json.loads, lines)]
    matrix = np.load(io.BytesIO(vectors_npy)).astype(np.float32)
    if len(items) != len(matrix):
        raise ValueError(f"snapshot has {len(items)} items but {len(matrix)} vectors")
    return items, matrix


async def _read(location: str, name: str) -> bytes:
    """A file of the snapshot: a local directory, or gs://bucket/prefix read with the service's own credentials."""
    if not location.startswith("gs://"):
        return (Path(location) / name).read_bytes()
    bucket, _, prefix = location[5:].partition("/")
    async with httpx.AsyncClient(timeout=120) as client:
        token = (await client.get(METADATA_TOKEN, headers={"Metadata-Flavor": "Google"})).json()["access_token"]
        object_name = f"{prefix.strip('/')}/{name}".lstrip("/")
        response = await client.get(
            f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{quote(object_name, safe='')}",
            params={"alt": "media"}, headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()
        return response.content


class MemoryRetriever:
    def __init__(self, cosmos, source_collections: dict[str, str], collections: list[str], snapshot: str,
                 catch_up_seconds: float = 300):
        self.cosmos = cosmos
        self.source_collections = source_collections
        self.names = collections
        self.snapshot = snapshot
        self.catch_up_seconds = catch_up_seconds
        self.indexes: dict[str, Index] = {}
        self.status: dict[str, str] = {name: "not loaded" for name in collections}
        self._checked: dict[str, float] = {}
        self._busy: dict[str, asyncio.Lock] = {name: asyncio.Lock() for name in collections}
        self._floor: dict[str, float] = {}

    async def load(self) -> None:
        for name in self.names:
            try:
                started = time.monotonic()
                items, matrix = parse_snapshot(await _read(self.snapshot, f"{name}.jsonl.gz"),
                                               await _read(self.snapshot, f"{name}.npy"))
                self._floor[name] = max((i.ingested_at.timestamp() for i in items), default=0.0)
                self.indexes[name] = Index(items, matrix, self.source_collections)
                await self.catch_up(name)
                self.status[name] = f"{len(self.indexes[name])} items in memory"
                logger.info("collection %s: %s (%.1fs)", name, self.status[name], time.monotonic() - started)
            except Exception as exc:  # noqa: BLE001 - serve from Cosmos instead, visibly
                self.status[name] = f"snapshot failed ({type(exc).__name__}: {str(exc)[:100]}); using Cosmos"
                logger.exception("collection %s: could not load the snapshot; using Cosmos", name)

    async def catch_up(self, name: str) -> int:
        """Add what Cosmos holds from after the snapshot and is not in memory yet."""
        async with self._busy[name]:
            self._checked[name] = time.monotonic()
            index = self.indexes[name]
            known = {i.id for i in index.items}
            fresh = [(i, v) for i, v in await self.cosmos.items_since(name, self._floor[name]) if i.id not in known]
            if fresh:
                matrix = np.vstack([index.matrix, np.vstack([v for _, v in fresh]).astype(np.float32)])
                self.indexes[name] = Index(index.items + [i for i, _ in fresh], matrix, self.source_collections)
                logger.info("collection %s: %d items added since the snapshot", name, len(fresh))
            return len(fresh)

    async def search(self, vector: np.ndarray, since: datetime | None, collection: str | None, limit: int) -> list[Candidate]:
        index = self.indexes.get(collection or "")
        if index is None:
            return await self.cosmos.search(vector, since, collection, limit)
        if time.monotonic() - self._checked[collection] > self.catch_up_seconds and not self._busy[collection].locked():
            asyncio.create_task(self._safe_catch_up(collection))
        return await index.search(vector, since, collection, limit)

    async def _safe_catch_up(self, name: str) -> None:
        try:
            await self.catch_up(name)
        except Exception:  # noqa: BLE001 - the next query tries again
            logger.exception("collection %s: catch-up failed; serving what is in memory", name)
