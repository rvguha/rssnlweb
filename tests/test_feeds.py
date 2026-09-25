from datetime import UTC, datetime

import pytest

from qdrss.feeds import FeedError, parse_date, parse_feed

from .conftest import ATOM, NOW, RSS


def test_rss_items():
    items = parse_feed(RSS, "a", NOW)
    assert [i.id for i in items][:2] == ["a:a-1", "a:https://a.test/no-guid"]
    assert items[2].id.startswith("a:") and len(items[2].id) == 2 + 24
    assert items[0].content == "The Rust language ships version 2.0 & a new borrow checker."
    assert items[0].published_at == datetime(2026, 9, 21, 10, tzinfo=UTC)
    assert items[1].title == "Untitled bold"
    assert items[1].content == "Encoded body wins"
    assert items[1].published_at is None
    assert all(i.ingested_at == NOW and i.source_title == "Feed A" for i in items)


def test_hash_id_is_stable_across_reorder():
    first = parse_feed(RSS, "a", NOW)[2].id
    reordered = RSS.replace(b"<item><title>No guid", b"</channel></rss>", 0)
    assert parse_feed(reordered, "a", NOW)[2].id == first


def test_atom_prefers_alternate_link():
    (item,) = parse_feed(ATOM, "b", NOW)
    assert item.url == "https://b.test/python4"
    assert item.id == "b:urn:b:1"
    assert item.published_at == datetime(2026, 9, 22, 8, 30, tzinfo=UTC)


def test_same_guid_in_two_sources_is_two_items():
    assert parse_feed(RSS, "x", NOW)[0].id != parse_feed(RSS, "y", NOW)[0].id


def test_bad_feed_raises():
    with pytest.raises(FeedError):
        parse_feed(b"<html>not a feed</html>", "a", NOW)
    with pytest.raises(FeedError):
        parse_feed(b"<rss><channel><item>", "a", NOW)


@pytest.mark.parametrize(
    "raw",
    ["Mon, 21 Sep 2026 10:00:00 GMT", "2026-09-21T10:00:00Z", "2026-09-21T12:00:00+02:00"],
)
def test_parse_date(raw):
    assert parse_date(raw) == datetime(2026, 9, 21, 10, tzinfo=UTC)


def test_parse_date_bad():
    assert parse_date("yesterday") is None
    assert parse_date("") is None
