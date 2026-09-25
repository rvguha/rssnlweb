from datetime import timedelta

import httpx
import numpy as np

from qdrss.feeds import parse_feed
from qdrss.ingest import Source, refresh
from qdrss.store import Store

from .conftest import NOW, RSS, make_item


def test_insert_if_absent_keeps_first_version(store: Store):
    assert store.insert_missing([make_item("1", "v1")]) == 1
    assert store.insert_missing([make_item("1", "v2"), make_item("2", "n")]) == 1
    titles = {i.id: i.title for i in store.all_items()}
    assert titles == {"s:1": "v1", "s:2": "n"}


def test_items_survive_reopen(tmp_path):
    path = tmp_path / "db.sqlite"
    Store(path).insert_missing([make_item("1", "t")])
    assert [i.id for i in Store(path).all_items()] == ["s:1"]


def test_embedding_cache_roundtrip(store: Store):
    store.save_embeddings("m", {"a": np.array([1.0, 2.0], dtype=np.float32)})
    found = store.embeddings("m", ["a", "b"])
    assert list(found) == ["a"] and found["a"].tolist() == [1.0, 2.0]
    assert store.embeddings("other", ["a"]) == {}


class Server:
    """Scripted upstream: a list of (status, headers, body) responses in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, headers, body = self.responses.pop(0)
        return httpx.Response(status, headers=headers, content=body)


SOURCE = Source("a", "https://a.test/feed")


async def run(server: Server, store: Store, source: Source = SOURCE):
    async with httpx.AsyncClient(transport=httpx.MockTransport(server.handler)) as client:
        return (await refresh([source], store, client=client))[0]


async def test_conditional_get_then_304(store: Store):
    server = Server((200, {"ETag": '"v1"'}, RSS), (304, {}, b""))
    first = await run(server, store)
    assert first.new_items == 3 and first.changed
    second = await run(server, store)
    assert server.requests[1].headers["If-None-Match"] == '"v1"'
    assert second.new_items == 0 and not second.changed and second.error is None
    assert store.count() == 3


async def test_identical_body_200_is_unchanged(store: Store):
    server = Server((200, {}, RSS), (200, {}, RSS))
    await run(server, store)
    assert not (await run(server, store)).changed


async def test_new_item_appended_and_rolled_off_items_kept(store: Store):
    rolled = RSS.replace(b"<guid isPermaLink=\"false\">a-1</guid>", b"<guid>a-new</guid>")
    server = Server((200, {}, RSS), (200, {}, rolled))
    await run(server, store)
    out = await run(server, store)
    assert out.new_items == 1
    assert store.count() == 4  # a-1 rolled off upstream but stays


async def test_failure_keeps_validators_and_items(store: Store):
    server = Server((200, {"ETag": '"v1"'}, RSS), (500, {}, b""), (304, {}, b""))
    await run(server, store)
    failed = await run(server, store)
    assert failed.error == "HTTP 500" and store.count() == 3
    assert store.source_state("a")["etag"] == '"v1"'
    await run(server, store)
    assert server.requests[2].headers["If-None-Match"] == '"v1"'
    assert store.source_state("a")["last_error"] is None


async def test_malformed_body_does_not_update_validators(store: Store):
    server = Server((200, {"ETag": '"bad"'}, b"<html>oops</html>"), (200, {}, RSS))
    bad = await run(server, store)
    assert bad.error and "not XML" in bad.error or "neither" in bad.error
    assert "If-None-Match" not in (await run(server, store), server.requests[1])[1].headers
    assert store.count() == 3


async def test_one_failing_source_does_not_block_another(store: Store):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "bad.test":
            raise httpx.ConnectError("refused")
        return httpx.Response(200, content=RSS)

    sources = [Source("bad", "https://bad.test/f"), Source("good", "https://good.test/f")]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcomes = await refresh(sources, store, client=client)
    assert outcomes[0].error and outcomes[1].new_items == 3


def test_ingested_at_is_first_seen(store: Store):
    early = parse_feed(RSS, "a", NOW - timedelta(days=3))
    late = parse_feed(RSS, "a", NOW)
    store.insert_missing(early)
    store.insert_missing(late)
    assert all(i.ingested_at == NOW - timedelta(days=3) for i in store.all_items())


PAGE2 = RSS.replace(b"a-1", b"a-old").replace(b"<channel>", b"<channel>", 1)


async def test_first_fetch_follows_archive_pages(store: Store):
    paged = RSS.replace(
        b"<channel>",
        b'<channel><atom:link xmlns:atom="http://www.w3.org/2005/Atom" rel="next" '
        b'href="https://a.test/feed?page=2"/>',
    )
    server = Server((200, {}, paged), (200, {}, PAGE2), (200, {}, paged))
    first = await run(server, store)
    assert first.new_items == 4  # 3 on page one, 1 more (a-old) on page two
    assert str(server.requests[1].url) == "https://a.test/feed?page=2"
    await run(server, store)
    assert len(server.requests) == 3  # a later fetch does not walk the archive again


def test_sectioned_manifest(tmp_path):
    from qdrss.ingest import load_sources

    path = tmp_path / "s.yaml"
    path.write_text("ai:\n- name: a\n  url: https://a\nhistory:\n- name: b\n  url: https://b\n")
    assert [(s.name, s.collection) for s in load_sources(path)] == [("a", "ai"), ("b", "history")]
    path.write_text("- name: a\n  url: https://a\n")
    assert load_sources(path)[0].collection == "default"
    path.write_text("Bad Name:\n- name: a\n  url: https://a\n")
    import pytest

    with pytest.raises(ValueError):
        load_sources(path)
