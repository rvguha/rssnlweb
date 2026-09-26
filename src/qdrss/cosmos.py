"""Azure Cosmos DB for NoSQL as the item + vector store.

One container, `items`, partitioned by collection, holds each episode with its
embedding; Cosmos's DiskANN index answers the nearest-neighbour query with the
date and collection filters applied inside the query. A second container,
`sources`, holds fetch validators. The web app keeps nothing in memory.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
from azure.cosmos import PartitionKey, exceptions
from azure.cosmos.aio import CosmosClient

from .models import Candidate, Item

logger = logging.getLogger(__name__)

DIMENSIONS = 1536
SNIPPET_CHARS = 1500  # what the ranker sees
CONTENT_CHARS = 8000  # what gets embedded

VECTOR_POLICY = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": DIMENSIONS,
        }
    ]
}
INDEXING_POLICY = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    # The vector must be excluded from the ordinary index or every write pays
    # to index 1536 numbers; it is served by the vector index below.
    "excludedPaths": [{"path": "/embedding/*"}, {"path": "/content/?"}, {"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
}


def doc_id(item_id: str) -> str:
    # Item ids carry the source url, and Cosmos ids may not contain / ? # or \.
    return hashlib.sha256(item_id.encode()).hexdigest()[:32]


class CosmosStore:
    def __init__(self, endpoint: str, key: str, database: str = "qdrss"):
        self.client = CosmosClient(endpoint, credential=key)
        self.database_name = database
        self._db = None
        self._items = None
        self._sources = None

    async def setup(self) -> None:
        """Create database and containers if they do not exist. Idempotent."""
        self._db = await self.client.create_database_if_not_exists(self.database_name)
        self._items = await self._db.create_container_if_not_exists(
            id="items",
            partition_key=PartitionKey(path="/collection"),
            indexing_policy=INDEXING_POLICY,
            vector_embedding_policy=VECTOR_POLICY,
        )
        self._sources = await self._db.create_container_if_not_exists(
            id="sources", partition_key=PartitionKey(path="/name")
        )

    async def close(self) -> None:
        await self.client.close()

    # --- items ---------------------------------------------------------------

    async def insert_missing(self, items: list[Item]) -> int:
        """Create documents for items not yet stored. Existing ones are untouched."""
        inserted = 0
        for item in items:
            try:
                await self._items.create_item(_document(item))
                inserted += 1
            except exceptions.CosmosResourceExistsError:
                continue
        return inserted

    async def items_without_embedding(self, limit: int = 500) -> list[Item]:
        query = (
            "SELECT TOP @limit * FROM c WHERE NOT IS_DEFINED(c.embedding)"
        )
        rows = self._items.query_items(query, parameters=[{"name": "@limit", "value": limit}])
        return [_item(row) async for row in rows]

    async def save_embeddings(self, model: str, vectors: dict[str, np.ndarray]) -> None:
        for item_id, vector in vectors.items():
            row = await self._read(item_id)
            if row is None:
                continue
            row["embedding"] = [float(x) for x in vector]
            row["embedding_model"] = model
            await self._items.upsert_item(row)

    async def _read(self, item_id: str) -> dict[str, Any] | None:
        rows = self._items.query_items(
            "SELECT * FROM c WHERE c.id = @id", parameters=[{"name": "@id", "value": doc_id(item_id)}]
        )
        async for row in rows:
            return row
        return None

    async def search(
        self,
        vector: np.ndarray,
        since: datetime | None,
        collection: str | None,
        limit: int,
    ) -> list[Candidate]:
        clauses = ["IS_DEFINED(c.embedding)"]
        params: list[dict[str, Any]] = [
            {"name": "@k", "value": limit},
            {"name": "@v", "value": [float(x) for x in vector]},
        ]
        if since is not None:
            clauses.append("c.ingested_ts > @since")
            params.append({"name": "@since", "value": since.timestamp()})
        if collection:
            clauses.append("c.collection = @collection")
            params.append({"name": "@collection", "value": collection})
        query = (
            "SELECT TOP @k c.item_id, c.source, c.source_title, c.url, c.title, c.snippet, "
            "c.published_at, c.published_raw, c.ingested_at, c.collection, "
            "VectorDistance(c.embedding, @v) AS score "
            f"FROM c WHERE {' AND '.join(clauses)} "
            "ORDER BY VectorDistance(c.embedding, @v)"
        )
        rows = self._items.query_items(query, parameters=params)
        return [Candidate(_item(row), "vector", float(row["score"])) async for row in rows]

    async def count(self) -> int:
        rows = self._items.query_items("SELECT VALUE COUNT(1) FROM c")
        async for n in rows:
            return int(n)
        return 0

    async def counts_by_collection(self) -> dict[str, int]:
        rows = self._items.query_items(
            "SELECT c.collection AS k, COUNT(1) AS n FROM c GROUP BY c.collection"
        )
        return {row["k"]: int(row["n"]) async for row in rows}

    # --- sources -------------------------------------------------------------

    async def source_state(self, name: str) -> dict[str, str | None]:
        try:
            row = await self._sources.read_item(name, partition_key=name)
        except exceptions.CosmosResourceNotFoundError:
            return {}
        return {k: row.get(k) for k in ("url", "etag", "last_modified", "sha256", "last_fetch", "last_error")}

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
        await self._sources.upsert_item(
            {
                "id": name,
                "name": name,
                "url": url,
                "etag": etag if error is None else previous.get("etag"),
                "last_modified": last_modified if error is None else previous.get("last_modified"),
                "sha256": sha256 if error is None else previous.get("sha256"),
                "last_fetch": datetime.now(UTC).isoformat(),
                "last_error": error,
            }
        )

    async def all_sources(self) -> list[dict[str, str | None]]:
        rows = self._sources.query_items(
            "SELECT c.name, c.url, c.last_fetch, c.last_error FROM c ORDER BY c.name"
        )
        return [dict(row) async for row in rows]


def _document(item: Item, vector: np.ndarray | None = None, model: str | None = None) -> dict:
    doc: dict[str, Any] = {
        "id": doc_id(item.id),
        "item_id": item.id,
        "source": item.source,
        "source_title": item.source_title,
        "collection": item.collection or "default",
        "url": item.url,
        "title": item.title,
        "snippet": item.content[:SNIPPET_CHARS],
        "content": item.content[:CONTENT_CHARS],
        "published_at": item.published_at.isoformat() if item.published_at else None,
        "published_raw": item.published_raw,
        "ingested_at": item.ingested_at.isoformat(),
        "ingested_ts": item.ingested_at.timestamp(),
    }
    if vector is not None:
        doc["embedding"] = [float(x) for x in vector]
        doc["embedding_model"] = model
    return doc


def _item(row: dict[str, Any]) -> Item:
    return Item(
        id=row["item_id"],
        source=row["source"],
        source_title=row.get("source_title", ""),
        url=row.get("url", ""),
        title=row.get("title", ""),
        content=row.get("content") or row.get("snippet", ""),
        published_at=datetime.fromisoformat(row["published_at"]) if row.get("published_at") else None,
        published_raw=row.get("published_raw", ""),
        ingested_at=datetime.fromisoformat(row["ingested_at"]),
        collection=row.get("collection", ""),
    )
