"""Backfill New York Times podcast history from the NYT Archive API.

The NYT's podcast RSS feeds carry only recent episodes (The Ezra Klein Show: 5 of
538), and Apple's directory mirrors them. Every episode also has a page on
nytimes.com, and the Archive API returns each month's pages, so one call per
month recovers the history: title, summary, date and link.

- Each page is filed under the nytimes source it belongs to, by URL pattern, or
  under `nyt_archive` when no show matches. Transcript pages are skipped: they
  duplicate an episode.
- An episode RSS already gave us is skipped, matched on title (or URL). Not by
  date: feeds keep a few old "evergreen" items (Ezra Klein's 2021 trailer, Daily
  Sunday Reads), so "older than the earliest RSS item" skipped years RSS lacks.
- `ingested_at` is the publication date, not now. QDRSS feeds show what was
  ingested within the window; stamping a decade of episodes "now" would present
  them as new in every nytimes feed for a week. Back-dated, they are reached with
  a far-back `since` (the UI's "Include archive" option).

Needs NYT_API_KEY in .env (developer.nytimes.com, Archive API enabled). The
free tier allows 5 calls a minute; responses are cached under data/nyt_archive
so a rerun only fetches what is missing.

    .venv/bin/python scripts/backfill_nyt_archive.py [--start 2006-01] [--end 2026-10]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from qdrss.app import build_services
from qdrss.feeds import parse_date
from qdrss.models import Item

CACHE = Path("data/nyt_archive")
COLLECTION = "nytimes"
FALLBACK = "nyt_archive"
FALLBACK_TITLE = "New York Times podcasts (archive)"
# Seconds between uncached calls: the free tier allows 5 a minute.
SPACING = 12.5

# URL fragment -> source, first match wins. Specific shows before broad ones.
RULES: list[tuple[str, str]] = [
    ("/podcasts/the-daily/", "nyt_the_daily"),
    ("ezra-klein", "nyt_the_ezra_klein_show"),
    ("hard-fork", "hard_fork"),
    ("run-up", "nyt_the_run_up"),
    ("modern-love", "nyt_modern_love"),
    ("popcast", "nyt_popcast"),
    ("book-review", "nyt_the_book_review"),
    ("the-interview", "nyt_the_interview"),
    ("matter-of-opinion", "nyt_matter_of_opinion"),
    ("interesting-times", "nyt_interesting_times"),
    ("ross-douthat", "nyt_interesting_times"),
    ("cannonball", "nyt_cannonball_with_wesley_morris"),
    ("still-processing", "nyt_still_processing"),
    ("sway-", "nyt_sway"),
    ("the-argument", "nyt_the_argument"),
    ("wirecutter", "nyt_the_wirecutter_show"),
    ("the-headlines", "nyt_the_headlines"),
    ("first-person", "nyt_first_person"),
    ("the-choice", "nyt_the_choice"),
    ("climate-forward", "nyt_climate_forward_podcast"),
    ("rabbit-hole", "nyt_rabbit_hole"),
    ("caliphate", "nyt_caliphate"),
    ("1619", "nyt_1619"),
]


def is_episode(doc: dict) -> bool:
    url = (doc.get("web_url") or "").lower()
    headline = ((doc.get("headline") or {}).get("main") or "").lower()
    if "transcript" in url or headline.startswith("transcript"):
        return False
    return (
        "/podcasts/" in url
        or "podcast" in url.rsplit("/", 1)[-1]
        or (doc.get("type_of_material") or "").lower() == "podcast"
    )


def source_for(doc: dict) -> str:
    url = (doc.get("web_url") or "").lower()
    for fragment, source in RULES:
        if fragment in url:
            return source
    # The Headlines' pages carry no show name in the URL; their summary does.
    if (doc.get("abstract") or "").startswith("Hear the news in five minutes"):
        return "nyt_the_headlines"
    return FALLBACK


def norm(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    year, month = map(int, start.split("-"))
    last = tuple(map(int, end.split("-")))
    out = []
    while (year, month) <= last:
        out.append((year, month))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def fetch_month(year: int, month: int, key: str) -> list[dict]:
    path = CACHE / f"{year}-{month:02d}.json"
    if path.is_file():
        return json.loads(path.read_text())
    url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={key}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                docs = json.load(response)["response"]["docs"]
            break
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 5:
                raise
            time.sleep(60)
    time.sleep(SPACING)
    episodes = [doc for doc in docs if is_episode(doc)]
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episodes))
    return episodes


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2006-01")
    parser.add_argument("--end", default=datetime.now(UTC).strftime("%Y-%m"))
    args = parser.parse_args()
    load_dotenv()
    key = os.environ["NYT_API_KEY"]
    logging.basicConfig(level=logging.WARNING)

    services = build_services()
    await services.setup()
    store = services.store
    try:
        container = store._items  # a one-off maintenance script reads the container directly
        titles: dict[str, str] = {}
        held: set[str] = set()  # normalized titles and urls of episodes already stored
        rows = container.query_items(
            "SELECT c.source, c.source_title, c.title, c.url FROM c WHERE c.kind = 'item'",
            partition_key=COLLECTION,
        )
        async for row in rows:
            titles.setdefault(row["source"], row.get("source_title") or row["source"])
            if row.get("title"):
                held.add(norm(row["title"]))
            if row.get("url"):
                held.add(row["url"])

        added = skipped = 0
        by_source: dict[str, int] = {}
        for year, month in month_range(args.start, args.end):
            items = []
            for doc in fetch_month(year, month, key):
                source = source_for(doc)
                published = parse_date(doc.get("pub_date") or "")
                if published is None:
                    continue
                headline = (doc.get("headline") or {}).get("main") or ""
                if doc["web_url"] in held or (headline and norm(headline) in held):
                    skipped += 1  # RSS already gave us this episode
                    continue
                content = "\n".join(
                    part for part in (doc.get("abstract"), doc.get("lead_paragraph")) if part
                )
                items.append(Item(
                    id=f"{source}:{doc['web_url']}",
                    source=source,
                    source_title=titles.get(source, FALLBACK_TITLE),
                    url=doc["web_url"],
                    title=headline,
                    content=content,
                    published_at=published,
                    published_raw=doc.get("pub_date") or "",
                    ingested_at=published,
                    collection=COLLECTION,
                ))
                by_source[source] = by_source.get(source, 0) + 1
            if items:
                added += await store.insert_missing(items)
            print(f"{year}-{month:02d}: {len(items):4} episodes  (stored so far {added})",
                  flush=True)
        print(f"stored {added} new archive episodes; skipped {skipped} already covered by RSS")
        for source, count in sorted(by_source.items(), key=lambda x: -x[1])[:25]:
            print(f"  {source:40} {count}")
        print("embedded:", await services.embed_missing(), flush=True)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
