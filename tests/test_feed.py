import xml.etree.ElementTree as ET
from datetime import timedelta

import pytest

from qdrss.feed import NS, FeedRequest, evaluate, render_rss
from qdrss.index import build_index
from qdrss.models import Candidate
from qdrss.providers import HashEmbeddings, KeywordRanker
from qdrss.rank import RankingError, classify
from qdrss.store import Store

from .conftest import NOW, make_item

OPTS = {"candidate_count": 3, "batch_size": 2, "window_days": 7}


async def populated(store: Store):
    old = NOW - timedelta(days=30)
    store.insert_missing(
        [
            make_item(f"old{i}", "rust compiler rust compiler", "rust", ingested=old)
            for i in range(5)
        ]
        + [
            make_item("new1", "rust release", "the rust team", ingested=NOW),
            make_item("new2", "rust notes", "compiler", ingested=NOW),
            make_item("new3", "gardening", "tomatoes", ingested=NOW - timedelta(days=1)),
        ]
    )
    return await build_index(store, HashEmbeddings())


async def test_date_mask_applies_before_top_k(store: Store):
    index = await populated(store)
    req = FeedRequest("rust compiler", since=NOW - timedelta(days=7), limit=10, threshold="relevant")
    matches = await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS)
    ids = [m.item.id for m in matches]
    # Five older items are the closest vectors and would fill a top-3 pool; the
    # mask keeps them out so the two new rust items are what comes back.
    assert ids == ["s:new1", "s:new2"]
    assert {m.category for m in matches} == {"relevant", "strong"}


async def test_strong_threshold_and_limit(store: Store):
    index = await populated(store)
    req = FeedRequest("rust compiler", since=NOW - timedelta(days=7), limit=1, threshold="strong")
    matches = await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS)
    assert [m.item.id for m in matches] == ["s:new2"]


async def test_default_window_when_no_since(store: Store):
    index = await populated(store)
    req = FeedRequest("rust", since=None, limit=10, threshold="relevant")
    matches = await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS)
    assert all(m.item.ingested_at > NOW - timedelta(days=7) for m in matches)


async def test_newest_first_with_id_tiebreak(store: Store):
    index = await populated(store)
    req = FeedRequest("rust", since=None, limit=10, threshold="relevant")
    matches = await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS)
    keys = [(-m.item.ingested_at.timestamp(), m.item.id) for m in matches]
    assert keys == sorted(keys)


async def test_no_matches_is_empty_not_error(store: Store):
    index = await populated(store)
    req = FeedRequest("quantum chromodynamics", since=None, limit=10, threshold="relevant")
    assert await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS) == []


async def test_ranker_failure_raises():
    class Broken:
        async def structured(self, instruction, payload):
            raise TimeoutError("slow")

    with pytest.raises(RankingError):
        await classify("q", [Candidate(make_item("1", "t"), "vector", 1.0)], Broken())


async def test_ranker_output_validation():
    class Odd:
        async def structured(self, instruction, payload):
            return {
                "results": [
                    {"i": 0, "m": "STRONG", "why": "ok"},
                    {"i": 0, "m": "relevant"},  # duplicate
                    {"i": 9, "m": "strong"},  # unknown
                    {"i": 1, "m": "maybe"},  # bad category
                    "junk",
                ]
            }

    cands = [Candidate(make_item(str(i), "t"), "vector", 1.0) for i in range(2)]
    matches = await classify("q", cands, Odd())
    assert [(m.item.id, m.category) for m in matches] == [("s:0", "strong")]


async def test_render_rss_escapes_and_is_stable():
    item = make_item("x", 'Tom & Jerry <3', "a & b", ingested=NOW)
    from qdrss.models import Match

    req = FeedRequest("cats & dogs", since=NOW, limit=5, threshold="relevant")
    one = render_rss(req, [Match(item, "strong", "why")], "http://h")
    two = render_rss(req, [Match(item, "strong", "why")], "http://h")
    root = ET.fromstring(one)
    node = root.find("channel/item")
    assert node.findtext("title") == "Tom & Jerry <3"
    assert node.find("guid").get("isPermaLink") == "false"
    assert node.findtext("guid") == "s:x"
    assert node.findtext("pubDate").endswith("+0000")
    assert node.findtext(f"{{{NS}}}category") == "strong"
    assert node.findtext(f"{{{NS}}}ingestedAt") == NOW.isoformat()
    assert "since=" in root.findtext("channel/link")
    assert one.split(b"<lastBuildDate>")[0] == two.split(b"<lastBuildDate>")[0]


async def test_collection_mask(store: Store):
    store.insert_missing(
        [
            make_item("a1", "rust news", "rust", ingested=NOW, source="ai_show"),
            make_item("h1", "rust in history", "rust", ingested=NOW, source="hist_show"),
            make_item("x1", "rust orphan", "rust", ingested=NOW, source="gone"),
        ]
    )
    index = await build_index(
        store, HashEmbeddings(), {"ai_show": "ai", "hist_show": "history"}
    )
    async def ids(collection):
        req = FeedRequest("rust", since=None, limit=10, threshold="relevant", collection=collection)
        return sorted(m.item.id for m in await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS))
    assert await ids("ai") == ["ai_show:a1"]
    assert await ids("history") == ["hist_show:h1"]
    assert await ids(None) == ["ai_show:a1", "gone:x1", "hist_show:h1"]
    req = FeedRequest("rust", since=None, limit=10, threshold="relevant", collection="ai")
    assert b"collection=ai" in render_rss(req, [], "http://h")
