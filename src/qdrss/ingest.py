"""Fetch every source in sources.yaml and store the items we have not seen.

Conditional GETs (ETag / Last-Modified) avoid the body when the server honours
them; a body hash catches servers that answer 200 regardless. One failing
source never blocks the others, and a failure leaves that source's stored
items and validators as they were.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from .feeds import FeedError, next_page, parse_feed
from .store import Store

StoreLike = Store  # any object with the async Store interface (see cosmos.CosmosStore)

logger = logging.getLogger(__name__)
TIMEOUT = httpx.Timeout(30.0)
MAX_BYTES = 64 * 1024 * 1024  # some daily shows ship 50 MB feeds (Good Morning Football)
MAX_ARCHIVE_PAGES = 200


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    collection: str = "default"


@dataclass
class Outcome:
    source: Source
    new_items: int = 0
    changed: bool = False
    error: str | None = None


def load_sources(path: Path) -> list[Source]:
    """A manifest is either a list of sources, or a mapping of collection -> list.

    The section a source sits in is its collection, which the UI shows as a tab
    and `/feed.xml?collection=` filters on. A bare list is the "default" collection.
    """
    data = yaml.safe_load(path.read_text()) or {}
    sections: dict[str, list] = data if isinstance(data, dict) else {"default": data}
    sources: list[Source] = []
    for collection, entries in sections.items():
        if not re.fullmatch(r"[a-z0-9_-]+", str(collection)):
            raise ValueError(f"{path}: collection name {collection!r} must be [a-z0-9_-]")
        for entry in entries or []:
            if not isinstance(entry, dict) or "name" not in entry or "url" not in entry:
                raise ValueError(f"{path}: each source needs name and url, got {entry!r}")
            sources.append(Source(str(entry["name"]), str(entry["url"]), str(collection)))
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate source names")
    return sources


async def refresh(
    sources: list[Source],
    store: Store,
    *,
    concurrency: int = 6,
    client: httpx.AsyncClient | None = None,
) -> list[Outcome]:
    owned = client is None
    client = client or httpx.AsyncClient(
        follow_redirects=True, timeout=TIMEOUT, headers={"User-Agent": "qdrss/0.1"}
    )
    limit = asyncio.Semaphore(concurrency)
    try:
        return list(await asyncio.gather(*(_fetch_one(s, store, client, limit) for s in sources)))
    finally:
        if owned:
            await client.aclose()


async def _fetch_one(
    source: Source, store: Store, client: httpx.AsyncClient, limit: asyncio.Semaphore
) -> Outcome:
    previous = await store.source_state(source.name)
    headers: dict[str, str] = {}
    if previous.get("etag"):
        headers["If-None-Match"] = previous["etag"]  # type: ignore[index]
    if previous.get("last_modified"):
        headers["If-Modified-Since"] = previous["last_modified"]  # type: ignore[index]

    async with limit:
        try:
            response = await client.get(source.url, headers=headers)
        except httpx.HTTPError as exc:
            return await _failed(store, source, f"{type(exc).__name__}: {exc}")

    if response.status_code == 304:
        await store.remember_source(source.name, source.url, **_validators(response, previous))
        return Outcome(source)
    if response.status_code >= 400:
        return await _failed(store, source, f"HTTP {response.status_code}")
    body = response.content
    if len(body) > MAX_BYTES:
        return await _failed(store, source, f"{len(body)} bytes exceeds the limit")
    if not body.strip():
        return await _failed(store, source, "empty response")

    digest = hashlib.sha256(body).hexdigest()
    validators = _validators(response, previous, digest)
    if previous.get("sha256") == digest:
        await store.remember_source(source.name, source.url, **validators)
        return Outcome(source)

    # Parse before touching the store: a malformed body must not update validators,
    # or the next fetch would 304 and we would never see a corrected feed.
    # Parsing a 50 MB feed is seconds of CPU; keep it off the event loop so
    # feed fetches being served meanwhile are not stalled.
    try:
        items = _tag(await asyncio.to_thread(parse_feed, body, source.name, datetime.now(UTC)), source)
    except FeedError as exc:
        return await _failed(store, source, str(exc))

    inserted = await store.insert_missing(items)
    # First sight of a source: walk its RFC 5005 archive pages so we hold the
    # whole history. Later fetches only need page one, where new items appear.
    if not previous:
        inserted += await _archive(source, body, store, client, limit)
    await store.remember_source(source.name, source.url, **validators)
    logger.info("%s: %d items, %d new", source.name, len(items), inserted)
    return Outcome(source, new_items=inserted, changed=True)


async def _archive(
    source: Source,
    first_page: bytes,
    store: Store,
    client: httpx.AsyncClient,
    limit: asyncio.Semaphore,
) -> int:
    inserted = 0
    seen: set[str] = set()
    url = next_page(first_page)
    while url and url not in seen and len(seen) < MAX_ARCHIVE_PAGES:
        seen.add(url)
        async with limit:
            try:
                response = await client.get(url)
            except httpx.HTTPError as exc:
                logger.warning("%s: archive page %s: %s", source.name, url, exc)
                break
        if response.status_code != 200 or not response.content.strip():
            break
        try:
            items = _tag(
                await asyncio.to_thread(parse_feed, response.content, source.name, datetime.now(UTC)),
                source,
            )
        except FeedError as exc:
            logger.warning("%s: archive page %s: %s", source.name, url, exc)
            break
        inserted += await store.insert_missing(items)
        url = next_page(response.content)
    if seen:
        logger.info("%s: %d archive pages, %d items", source.name, len(seen), inserted)
    return inserted


async def _failed(store: Store, source: Source, error: str) -> Outcome:
    logger.warning("%s: %s", source.name, error)
    await store.remember_source(source.name, source.url, error=error)
    return Outcome(source, error=error)


def _tag(items: list, source: Source) -> list:
    from dataclasses import replace

    return [replace(item, collection=source.collection) for item in items]


def _validators(
    response: httpx.Response, previous: dict[str, str | None], digest: str | None = None
) -> dict[str, str | None]:
    return {
        "etag": response.headers.get("etag") or previous.get("etag"),
        "last_modified": response.headers.get("last-modified") or previous.get("last_modified"),
        "sha256": digest or previous.get("sha256"),
    }
