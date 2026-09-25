"""In-memory vector index over every stored item, rebuilt on refresh.

The date and collection mask is applied before top-K: a feed's candidates are
the best *new* items, not the new items among the best overall.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from .models import Candidate, Item
from .providers import Embeddings
from .store import Store


class Index:
    def __init__(
        self, items: list[Item], matrix: np.ndarray, collections: dict[str, str] | None = None
    ):
        self.items = items
        self.matrix = matrix
        # source name -> collection, from the manifest. Items whose source has
        # dropped out of the manifest belong to no collection.
        self._collection = np.asarray(
            [(collections or {}).get(item.source, "") for item in items], dtype=object
        )
        self.ingested = np.asarray(
            [item.ingested_at.timestamp() for item in items], dtype=np.float64
        )

    def __len__(self) -> int:
        return len(self.items)

    def eligible(self, since: datetime | None, collection: str | None = None) -> np.ndarray:
        mask = np.ones(len(self.items), dtype=bool)
        if since is not None:
            mask &= self.ingested > since.timestamp()
        if collection:
            mask &= self._collection == collection
        return mask

    def search(self, vector: np.ndarray, mask: np.ndarray, limit: int) -> list[Candidate]:
        rows = np.flatnonzero(mask)
        if not len(rows):
            return []
        scores = self.matrix[rows] @ vector
        order = np.argsort(-scores, kind="stable")[:limit]
        return [Candidate(self.items[rows[i]], "vector", float(scores[i])) for i in order]


async def build_index(
    store: Store, embedder: Embeddings, collections: dict[str, str] | None = None
) -> Index:
    """Embed only items with no cached vector for this embedding model."""
    items = store.all_items()
    cached = store.embeddings(embedder.model, [item.id for item in items])
    missing = [item for item in items if item.id not in cached]
    for start in range(0, len(missing), 100):
        chunk = missing[start : start + 100]
        vectors = await embedder.embed([item.text[:8000] for item in chunk])
        fresh = {item.id: vectors[i] for i, item in enumerate(chunk)}
        store.save_embeddings(embedder.model, fresh)
        cached.update(fresh)
    if items:
        matrix = np.vstack([cached[item.id] for item in items]).astype(np.float32)
    else:
        matrix = np.zeros((0, 1), dtype=np.float32)
    return Index(items, matrix, collections)
