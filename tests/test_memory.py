"""In-memory collections: snapshot round trip, catch-up from the database, fallback."""

from datetime import UTC, datetime

import numpy as np

from qdrss.memory import MemoryRetriever, parse_snapshot, write_snapshot
from qdrss.models import Item


def item(n, day):
    return Item(id=f"s:{n}", source="s", source_title="S", url=f"http://x/{n}", title=f"t{n}", content=f"c{n}",
                published_at=None, published_raw="", ingested_at=datetime(2020, 1, day, tzinfo=UTC), collection="tal")


class FakeCosmos:
    def __init__(self, fresh=()):
        self.fresh, self.searched = list(fresh), 0

    async def items_since(self, collection, since_ts):
        return [(i, v) for i, v in self.fresh if i.ingested_at.timestamp() > since_ts]

    async def search(self, *args):
        self.searched += 1
        return []


def snapshot(tmp_path):
    items = [item(1, 1), item(2, 2)]
    write_snapshot(tmp_path, "tal", items, np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    return items


def test_round_trip(tmp_path):
    snapshot(tmp_path)
    items, matrix = parse_snapshot((tmp_path / "tal.jsonl.gz").read_bytes(), (tmp_path / "tal.npy").read_bytes())
    assert [i.id for i in items] == ["s:1", "s:2"] and items[0].ingested_at.tzinfo and matrix.dtype == np.float32


async def test_searches_memory_and_adds_what_came_after_the_snapshot(tmp_path):
    snapshot(tmp_path)
    fresh = [(item(3, 9), np.asarray([0.8, 0.6], dtype=np.float32)), (item(2, 2), np.asarray([0, 1], dtype=np.float32))]
    cosmos = FakeCosmos(fresh)
    r = MemoryRetriever(cosmos, {"s": "tal"}, ["tal"], str(tmp_path))
    await r.load()
    assert len(r.indexes["tal"]) == 3                        # s:2 is not added twice
    got = await r.search(np.asarray([0.8, 0.6], dtype=np.float32), None, "tal", 2)
    assert [c.item.id for c in got][0] == "s:3" and cosmos.searched == 0


async def test_other_collections_and_failed_snapshots_use_cosmos(tmp_path):
    cosmos = FakeCosmos()
    r = MemoryRetriever(cosmos, {"s": "tal"}, ["tal"], str(tmp_path / "missing"))
    await r.load()
    assert "using Cosmos" in r.status["tal"]
    await r.search(np.zeros(2, dtype=np.float32), None, "tal", 5)
    await r.search(np.zeros(2, dtype=np.float32), None, None, 5)
    assert cosmos.searched == 2
