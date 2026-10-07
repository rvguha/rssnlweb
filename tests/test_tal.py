import asyncio
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from qdrss import tal
from qdrss.feed import NS, render_rss, FeedRequest
from qdrss.ingest import load_sources
from qdrss.models import Match
from qdrss.store import Store


def paragraphs(start: float, count: int, words: int = 40, speaker: str = ""):
    head = f"<h4>{speaker}</h4>" if speaker else ""
    body = "".join(f'<p begin="x" data-timestamp="{start + i * 10}">{" ".join(f"w{start}{i}x{j}" for j in range(words))}</p>' for i in range(count))
    return f'<div class="host">{head}{body}</div>'


def transcript() -> str:
    return (
        '<meta property="og:title" content="899: Reaching &amp; Out - This American Life" />'
        '<a href="/899/reaching-out">episode</a><a href="/899/transcript">t</a><main>'
        f'<div class="act" id="prologue"><h3>Prologue: Prologue</h3><div class="act-inner">{paragraphs(5, 4, 30, "Ira Glass")}</div></div>'
        f'<div class="act" id="act1"><h3>Act One: The Intern</h3><div class="act-inner">{paragraphs(100, 20, 40, "Ira Glass")}'
        f'{paragraphs(300, 3, 40, "Sayre")}</div></div>'
        f'<div class="act" id="credits"><h3>Credits</h3><div class="act-inner">{paragraphs(900, 2, 10)}</div></div>'
        "</main><footer></footer>"
    )


EPISODE_PAGE = (
    '<meta property="article:published_time" content="2026-09-25T11:30:23-04:00" />'
    '<div class="field field-name-field-radio-air-date">October 2, 2026</div>'
    '<a href="https://www.thisamericanlife.org/sites/default/files/audio/upload/schedule/clean/899.mp3">Download</a>'
)


def test_acts_and_their_labels():
    acts = tal.parse_transcript(transcript())
    assert [(a.id, a.label) for a in acts] == [("prologue", "Prologue"), ("act1", "Act One: The Intern"), ("credits", "Credits")]
    assert acts[0].paragraphs[0].start == 5.0 and acts[0].paragraphs[0].speaker == "Ira Glass"
    assert tal.act_label("Part One: Part One") == "Part One"
    assert tal.act_label("Act Two: The Witness") == "Act Two: The Witness"


def test_passages_stay_inside_an_act_and_carry_their_offset():
    acts = tal.parse_transcript(transcript())
    found = tal.passages(acts, window_words=280)
    for act in acts:
        inside = [p for p in found if p.act_id == act.id]
        assert inside[0].start == act.paragraphs[0].start
        assert all(a.end == b.start for a, b in zip(inside, inside[1:]))      # each ends where the next begins
    assert {p.act_id for p in found} == {"prologue", "act1", "credits"}
    assert all(len(p.text.split()) >= tal.MIN_TAIL_WORDS or p.act_id in ("credits", "prologue") for p in found)
    act1 = [p for p in found if p.act_id == "act1"]
    assert len(act1) > 1 and 250 <= len(act1[0].text.split()) <= 330                # about two minutes each


def test_speaker_is_named_when_it_changes():
    first = [p for p in tal.passages(tal.parse_transcript(transcript())) if p.act_id == "prologue"][0]
    assert first.text.startswith("Ira Glass: ") and first.text.count("Ira Glass:") == 1


def test_a_short_tail_joins_the_passage_before():
    # Two full passages of 300 words and a 50-word tail: the tail must not stand alone.
    words = [100] * 6 + [50]
    act = tal.Act("a", "A", tuple(tal.Paragraph(float(i * 10), "", " ".join(["w"] * n)) for i, n in enumerate(words)))
    sizes = [len(p.text.split()) for p in tal.passages([act], window_words=300)]
    assert sizes == [300, 350]
    # A tail that is long enough stands alone.
    words = [100] * 6 + [100]
    act = tal.Act("a", "A", tuple(tal.Paragraph(float(i * 10), "", " ".join(["w"] * n)) for i, n in enumerate(words)))
    assert [len(p.text.split()) for p in tal.passages([act], window_words=300)] == [300, 300, 100]


def test_episode_info():
    ep = tal.episode_info(899, transcript(), EPISODE_PAGE)
    assert ep.title == "899: Reaching & Out"
    assert ep.page == "https://www.thisamericanlife.org/899/reaching-out"
    assert ep.transcript == "https://www.thisamericanlife.org/899/transcript"
    assert ep.aired == datetime(2026, 10, 2, tzinfo=UTC)                           # the air date, not the page's publish time
    assert ep.audio.endswith("/clean/899.mp3")
    assert tal.episode_info(899, transcript(), EPISODE_PAGE, "https://feed/audio.mp3").audio == "https://feed/audio.mp3"


def test_section_items():
    ep = tal.episode_info(899, transcript(), EPISODE_PAGE)
    now = datetime(2026, 10, 7, tzinfo=UTC)
    items = tal.section_items(ep, transcript(), source="tal_this_american_life", collection="tal", ingested_at=now)
    assert len({i.id for i in items}) == len(items)
    first = items[0]
    assert first.id == "tal_this_american_life:899:prologue:001"
    assert first.title == "899: Reaching & Out — Prologue" and first.url == ep.page and first.collection == "tal"
    extra = json.loads(first.extra)
    assert extra["episode"] == 899 and extra["start"] == 5.0 and extra["audio"].endswith("899.mp3#t=5")
    assert extra["transcript"].endswith("/899/transcript#prologue")


def test_feed_carries_the_offset():
    ep = tal.episode_info(899, transcript(), EPISODE_PAGE)
    item = tal.section_items(ep, transcript(), source="s", collection="tal", ingested_at=datetime(2026, 10, 7, tzinfo=UTC))[3]
    xml = render_rss(FeedRequest("q", None, 5, "relevant", "tal"), [Match(item, "strong", "why")], "https://x.test")
    node = ET.fromstring(xml).find("channel/item")
    got = {c.tag.split("}")[-1]: c.text for c in node if c.tag.startswith("{" + NS)}
    assert got["episode"] == "899" and float(got["offsetSeconds"]) == json.loads(item.extra)["start"]
    assert got["act"] and got["audio"].endswith(f"#t={int(float(got['offsetSeconds']))}") and "transcript" in got


def test_ordinary_items_have_no_section_elements():
    from qdrss.models import Item
    item = Item("i", "s", "S", "https://u", "t", "c", None, "", datetime(2026, 1, 1, tzinfo=UTC))
    xml = render_rss(FeedRequest("q", None, 5, "relevant", None), [Match(item, "strong", "w")], "https://x.test")
    assert b"offsetSeconds" not in xml


def test_extra_survives_the_store(tmp_path):
    async def run():
        store = Store(tmp_path / "s.sqlite")
        item = tal.section_items(tal.episode_info(899, transcript(), EPISODE_PAGE), transcript(), source="s", collection="tal",
                                 ingested_at=datetime(2026, 10, 7, tzinfo=UTC))[0]
        assert await store.insert_missing([item]) == 1
        assert store.all_items()[0].extra == item.extra
    asyncio.run(run())


def test_an_old_database_gains_the_extra_column(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "old.sqlite")
    db.executescript("CREATE TABLE items (id TEXT PRIMARY KEY, source TEXT NOT NULL, source_title TEXT NOT NULL, url TEXT NOT NULL, "
                     "title TEXT NOT NULL, content TEXT NOT NULL, published_at TEXT, published_raw TEXT NOT NULL, ingested_at TEXT NOT NULL, "
                     "collection TEXT NOT NULL DEFAULT '')")
    db.close()
    assert "extra" in {r[1] for r in Store(tmp_path / "old.sqlite").db.execute("PRAGMA table_info(items)")}


def test_manifest_segmenter(tmp_path):
    good = tmp_path / "ok.yaml"
    good.write_text("tal:\n- name: a\n  url: https://x.test/f\n  segmenter: tal\n")
    assert load_sources(good)[0].segmenter == "tal"
    bad = tmp_path / "bad.yaml"
    bad.write_text("tal:\n- name: a\n  url: https://x.test/f\n  segmenter: nope\n")
    with pytest.raises(ValueError, match="unknown segmenter"):
        load_sources(bad)
    shipped = [s for s in load_sources(Path(__file__).resolve().parent.parent / "sources.yaml") if s.collection == "tal"]
    assert [(s.name, s.segmenter) for s in shipped] == [("tal_this_american_life", "tal")]


FEED = b"""<?xml version="1.0"?><rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"><channel>
<item><title>899: Reaching Out</title><itunes:episode>899</itunes:episode><enclosure url="https://feed/899.mp3" type="audio/mpeg"/></item>
<item><title>898: Next</title><enclosure url="https://feed/898.mp3" type="audio/mpeg"/></item></channel></rss>"""


def test_feed_episodes():
    assert tal.feed_episodes(FEED) == {899: "https://feed/899.mp3", 898: "https://feed/898.mp3"}


def test_expand_skips_an_episode_without_a_transcript():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/899/transcript":
            return httpx.Response(200, text=transcript())
        if path == "/899/reaching-out":
            return httpx.Response(200, text=EPISODE_PAGE)
        return httpx.Response(404, text="nope")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await tal.expand(FEED, "tal_this_american_life", "tal", client, asyncio.Semaphore(2))
    items = asyncio.run(run())
    assert items and {json.loads(i.extra)["episode"] for i in items} == {899}
    assert json.loads(items[0].extra)["audio"].startswith("https://feed/899.mp3#t=")      # the feed's audio is preferred
