"""Parse RSS 2.0 and Atom into Items. No network here."""

from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from .models import Item

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


class FeedError(ValueError):
    pass


def parse_feed(body: bytes, source: str, ingested_at: datetime) -> list[Item]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise FeedError(f"{source}: not XML: {exc}") from exc
    tag = _local(root.tag)
    if tag == "rss" or _child(root, "channel") is not None:
        return _rss(root, source, ingested_at)
    if tag == "feed":
        return _atom(root, source, ingested_at)
    raise FeedError(f"{source}: root element <{tag}> is neither rss nor feed")


def next_page(body: bytes) -> str:
    """RFC 5005 archive link: <atom:link rel="next" href=...>, on the channel or feed."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return ""
    container = _child(root, "channel")
    if container is None:
        container = root
    for link in _children(container, "link"):
        if link.get("rel") == "next" and link.get("href"):
            return link.get("href", "")
    return ""


def _rss(root: ET.Element, source: str, ingested_at: datetime) -> list[Item]:
    channel = _child(root, "channel")
    if channel is None:
        channel = root
    source_title = _clean(_text(channel, "title"))
    items: list[Item] = []
    for node in _children(channel, "item"):
        title = _clean(_text(node, "title"))
        content = _clean(_text(node, "encoded") or _text(node, "description"))
        link = _text(node, "link").strip()
        guid = _text(node, "guid").strip()
        raw_date = (_text(node, "pubDate") or _text(node, "date")).strip()
        url = link or (guid if guid.startswith(("http://", "https://")) else "")
        if not url:
            enclosure = _child(node, "enclosure")
            url = enclosure.get("url", "") if enclosure is not None else ""
        items.append(
            _item(source, source_title, guid, url, title, content, raw_date, ingested_at)
        )
    return items


def _atom(root: ET.Element, source: str, ingested_at: datetime) -> list[Item]:
    source_title = _clean(_text(root, "title"))
    items: list[Item] = []
    for entry in _children(root, "entry"):
        title = _clean(_text(entry, "title"))
        content = _clean(_text(entry, "content") or _text(entry, "summary"))
        link = _atom_link(entry)
        guid = _text(entry, "id").strip()
        raw_date = (_text(entry, "published") or _text(entry, "updated")).strip()
        items.append(
            _item(source, source_title, guid, link, title, content, raw_date, ingested_at)
        )
    return items


def _item(
    source: str,
    source_title: str,
    guid: str,
    url: str,
    title: str,
    content: str,
    raw_date: str,
    ingested_at: datetime,
) -> Item:
    # Identity: guid, else link, else a hash of title + raw date. Namespaced by
    # source because upstream guids are only unique within their own feed.
    key = guid or url or hashlib.sha256(f"{title}\n{raw_date}".encode()).hexdigest()[:24]
    return Item(
        id=f"{source}:{key}",
        source=source,
        source_title=source_title,
        url=url,
        title=title,
        content=content,
        published_at=parse_date(raw_date),
        published_raw=raw_date,
        ingested_at=ingested_at,
    )


def parse_date(value: str) -> datetime | None:
    """RFC 2822 (RSS) or ISO 8601 (Atom), to aware UTC. None if unparseable."""
    value = value.strip()
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _atom_link(entry: ET.Element) -> str:
    fallback = ""
    for link in _children(entry, "link"):
        href = link.get("href", "").strip()
        if not href:
            continue
        if link.get("rel", "alternate") == "alternate":
            return href
        fallback = fallback or href
    return fallback


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(node: ET.Element, name: str) -> ET.Element | None:
    for child in node:
        if _local(child.tag) == name:
            return child
    return None


def _children(node: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in node if _local(child.tag) == name]


def _text(node: ET.Element, name: str) -> str:
    child = _child(node, name)
    return (child.text or "") if child is not None else ""


def _clean(value: str) -> str:
    value = html.unescape(_TAG_RE.sub(" ", value))
    return _WS_RE.sub(" ", value).strip()
