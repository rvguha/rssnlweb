import xml.etree.ElementTree as ET
from datetime import timedelta

from starlette.testclient import TestClient

from qdrss.app import Services, create_app
from qdrss.config import Config
from qdrss.index import build_index
from qdrss.providers import HashEmbeddings, KeywordRanker

from .conftest import NOW, make_item


def make_services(tmp_path) -> Services:
    (tmp_path / "sources.yaml").write_text("ai:\n- name: s\n  url: https://a.test/f\n")
    config = Config(
        data_dir=tmp_path,
        sources_path=tmp_path / "sources.yaml",
        host="127.0.0.1",
        port=0,
        refresh_minutes=60,
        window_days=7,
        candidate_count=10,
        default_limit=20,
        max_limit=50,
        batch_size=10,
        fetch_concurrency=2,
        openrouter_key="",
        openrouter_base_url="",
        ranking_model="",
        embedding_model="",
        provider_sort="",
    )
    return Services(config, HashEmbeddings(), KeywordRanker())


async def test_feed_endpoint(tmp_path):
    services = make_services(tmp_path)
    services.store.insert_missing(
        [
            make_item("1", "rust release", "rust", ingested=NOW - timedelta(days=1)),
            make_item("2", "old rust", "rust", ingested=NOW - timedelta(days=30)),
        ]
    )
    services.index = await build_index(services.store, services.embedder, services.collections)
    services.index_built_at = NOW
    app = create_app(services)
    services.start = lambda: None  # no refresh loop in tests

    with TestClient(app) as client:
        assert client.get("/feed.xml").status_code == 400
        assert client.get("/feed.xml?q=rust&since=nope").status_code == 400
        assert client.get("/feed.xml?q=rust&threshold=huge").status_code == 400

        r = client.get("/feed.xml", params={"q": "rust", "since": (NOW - timedelta(days=7)).isoformat()})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/rss+xml")
        guids = [n.text for n in ET.fromstring(r.content).findall("channel/item/guid")]
        assert guids == ["s:1"]

        r = client.get("/feed.xml", params={"q": "rust", "since": "2020-01-01T00:00:00Z"})
        guids = [n.text for n in ET.fromstring(r.content).findall("channel/item/guid")]
        assert guids == ["s:1", "s:2"]

        assert client.get("/feed.xml?q=rust&collection=nope").status_code == 400
        r = client.get("/feed.xml", params={"q": "rust", "since": "2020-01-01T00:00:00Z", "collection": "ai"})
        assert len(ET.fromstring(r.content).findall("channel/item")) == 2
        assert client.get("/collections").json() == {"ai": {"sources": 1, "items": 2}}

        home = client.get("/")
        assert home.status_code == 200 and b"<form" in home.content

        health = client.get("/health").json()
        assert health["items"] == 2


async def test_503_before_index_and_on_ranker_failure(tmp_path):
    services = make_services(tmp_path)
    services.start = lambda: None
    app = create_app(services)
    with TestClient(app) as client:
        assert client.get("/feed.xml?q=rust").status_code == 503

    # lifespan exit closed that store; the ranker-failure case gets its own services
    (tmp_path / "second").mkdir()
    services = make_services(tmp_path / "second")
    services.start = lambda: None
    app = create_app(services)

    class Broken:
        async def structured(self, instruction, payload):
            raise RuntimeError("provider down")

    services.store.insert_missing([make_item("1", "rust", "rust", ingested=NOW)])
    services.index = await build_index(services.store, services.embedder, services.collections)
    services.index_built_at = NOW
    services.ranker = Broken()
    with TestClient(app) as client:
        r = client.get("/feed.xml?q=rust&since=2020-01-01T00:00:00Z")
        assert r.status_code == 503 and b"ranking unavailable" in r.content
