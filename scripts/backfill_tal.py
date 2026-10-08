"""Backfill This American Life history as passages with audio offsets.

The TAL feed carries the latest 15 episodes. Every episode has a timed transcript at
thisamericanlife.org/<n>/transcript and a page at /<n>/<slug> (air date, audio); see qdrss/tal.py.
One call pair per episode recovers it, from the newest episode down to #1.

- Each episode becomes about 30 passages (qdrss.tal.section_items), filed under the `tal` collection and the
  tal_this_american_life source the feed uses, with the same ids, so the daily feed ingest and this script
  never duplicate each other.
- `ingested_at` is the air date, not now (as for the NYT backfill): a feed over the default window shows what was
  ingested lately, and 900 episodes stamped "now" would flood it. Reach them with a far-back `since`.
- Pages are cached under data/tal_cache, so a rerun fetches only what is missing. Requests are spaced out: it is
  a few thousand pages from a small non-profit's site, fetched once.
- Episodes without a timed transcript are listed and skipped.
- A transcript page the site answers with a server error every time is taken from the Internet Archive's newest capture that
  carries the timed transcript (qdrss.tal.archived_transcript); episodes 57, 79, 86 and 375 were recovered this way.

    .venv/bin/python scripts/backfill_tal.py [--first 1] [--last 899] [--workers 3] [--episodes 57,79,86,375]

Run it again after a failure: cached episodes are skipped and the stored ones are not duplicated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from qdrss import tal
from qdrss.app import build_services

SOURCE = "tal_this_american_life"
COLLECTION = "tal"
CACHE = Path("data/tal_cache")
PAUSE = 0.4  # seconds between one worker's requests


async def pages(number: int, client: httpx.AsyncClient) -> tuple[str, str, str] | None:
    """(transcript page, episode page, audio URL) from the cache or the site; None when there is no timed transcript.
    A throttled or failed request raises and is not cached, so a rerun fetches it again."""
    path = CACHE / f"{number}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        return (cached["transcript"], cached["episode"], cached["audio"]) if cached["transcript"] else None
    for attempt in range(4):
        try:
            got = await tal.fetch_pages(number, client)
            break
        except httpx.HTTPError:
            if attempt == 3:
                raise
            await asyncio.sleep(5 * (attempt + 1))        # throttled or a server error: back off, then try again
    await asyncio.sleep(PAUSE)
    path.write_text(json.dumps({"transcript": got[0] if got else "", "episode": got[1] if got else "", "audio": got[2] if got else ""}))
    return got


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=0, help="newest episode; default: the newest in the feed")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--episodes", help="comma-separated episode numbers to fetch instead of the whole --first/--last range")
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)
    CACHE.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(follow_redirects=True, timeout=60, headers={"User-Agent": "qdrss/0.1"}) as client:
        feed = tal.feed_episodes((await client.get(f"{tal.SITE}/podcast/rss.xml")).content)
        last = args.last or max(feed)
        services = build_services()
        await services.setup()
        store = services.store
        sem = asyncio.Semaphore(args.workers)
        missing: list[int] = []
        failed: list[int] = []
        added = done = 0

        async def one(number: int) -> int:
            nonlocal done
            async with sem:
                try:
                    got = await pages(number, client)
                except httpx.HTTPError as exc:
                    done += 1
                    failed.append(number)
                    print(f"episode {number}: {exc}", flush=True)
                    return 0
            done += 1
            if got is None:
                missing.append(number)
                return 0
            transcript, episode_page, audio = got
            episode = tal.episode_info(number, transcript, episode_page, feed.get(number, "") or audio)
            items = tal.section_items(episode, transcript, source=SOURCE, collection=COLLECTION,
                                      ingested_at=episode.aired or datetime.now(UTC))
            return await store.insert_missing(items)

        try:
            numbers = (sorted({int(n) for n in args.episodes.split(",")}, reverse=True) if args.episodes
                       else list(range(last, args.first - 1, -1)))
            for start in range(0, len(numbers), 50):
                added += sum(await asyncio.gather(*(one(n) for n in numbers[start:start + 50])))
                print(f"{done}/{len(numbers)} episodes; {added} passages stored; {len(missing)} without a transcript", flush=True)
            print(f"no timed transcript: {sorted(missing)}")
            if failed:
                print(f"FAILED (not cached; run again to retry): {sorted(failed)}")
            if services.cosmos:
                print("embedded:", await services.embed_missing(), flush=True)
            else:
                print("stored in SQLite; the next refresh embeds them")
        finally:
            await store.close()


if __name__ == "__main__":
    asyncio.run(main())
