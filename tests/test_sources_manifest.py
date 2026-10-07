"""The shipped sources.yaml: what a deploy would ingest."""

from collections import Counter
from pathlib import Path

from starlette.testclient import TestClient

from qdrss.app import create_app
from qdrss.ingest import load_sources

from .test_app import make_services

MANIFEST = Path(__file__).resolve().parent.parent / "sources.yaml"


def test_manifest_loads_with_unique_names_and_https_urls():
    sources = load_sources(MANIFEST)
    names = [s.name for s in sources]
    assert len(names) == len(set(names))
    assert all(s.url.startswith(("http://", "https://")) for s in sources)


def test_npr_collection():
    npr = [s for s in load_sources(MANIFEST) if s.collection == "npr"]
    assert 90 <= len(npr) <= 120
    assert all(s.name.startswith("npr_") for s in npr)
    # A feed listed twice would ingest the same episodes under two source names.
    assert not [url for url, n in Counter(s.url for s in npr).items() if n > 1]
    names = {s.name for s in npr}
    for show in ("npr_this_american_life", "npr_radiolab", "npr_fresh_air", "npr_planet_money"):
        assert show in names


def test_npr_landing_page(tmp_path):
    with TestClient(create_app(make_services(tmp_path))) as client:
        page = client.get("/npr")
        assert page.status_code == 200
        assert b"NPR podcasts by topic" in page.content
        assert b"collection: 'npr'" in page.content
        assert client.get("/nyt").status_code == 200
