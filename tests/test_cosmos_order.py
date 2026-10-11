"""The Cosmos retriever returns candidates closest first: cosine VectorDistance is a similarity, larger is closer."""
import asyncio

from qdrss.cosmos import CosmosStore


def row(i, score):
    return {"item_id": f"s:{i}", "source": "s", "ingested_at": "2026-01-01T00:00:00+00:00", "score": score}


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def __aiter__(self):
        async def gen():
            for r in self.rows:
                yield r
        return gen()


def test_candidates_come_back_closest_first_when_partitions_are_merged():
    store = CosmosStore.__new__(CosmosStore)
    # Two partitions, each already ordered closest first, concatenated.
    store._items = type("C", (), {"query_items": lambda self, *a, **k: Rows([row(1, 0.9), row(2, 0.5), row(3, 0.8), row(4, 0.4)])})()
    got = asyncio.run(store.search(__import__("numpy").zeros(3), None, None, 3))
    assert [c.item.id for c in got] == ["s:1", "s:3", "s:2"]
