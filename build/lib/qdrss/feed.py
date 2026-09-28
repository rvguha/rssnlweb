"""One stateless fetch: q + since -> the newest qualifying items, as RSS 2.0."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Protocol
from urllib.parse import urlencode

import numpy as np

from .models import Candidate, Match
from .providers import Embeddings, Ranker
from .rank import classify


class Retriever(Protocol):
    async def search(
        self, vector: np.ndarray, since: datetime | None, collection: str | None, limit: int
    ) -> list[Candidate]: ...

NS = "https://qdrss.dev/ns"

# Feed readers fetch the same URL for months. Cache the query vector per
# embedding model so a fetch costs one embedding call ever, not one per poll.
_QUERY_VECTORS: dict[tuple[str, str], np.ndarray] = {}
_QUERY_CACHE_MAX = 5000


async def _query_vector(embedder: Embeddings, q: str) -> np.ndarray:
    key = (embedder.model, q)
    vector = _QUERY_VECTORS.get(key)
    if vector is None:
        vector = (await embedder.embed([q]))[0]
        if len(_QUERY_VECTORS) >= _QUERY_CACHE_MAX:
            _QUERY_VECTORS.pop(next(iter(_QUERY_VECTORS)))
        _QUERY_VECTORS[key] = vector
    return vector


@dataclass(frozen=True)
class FeedRequest:
    q: str
    since: datetime | None
    limit: int
    threshold: str  # "relevant" (includes strong) | "strong"
    collection: str | None = None  # manifest section; None = every source


async def evaluate(
    request: FeedRequest,
    retriever: Retriever,
    embedder: Embeddings,
    ranker: Ranker,
    *,
    candidate_count: int,
    batch_size: int,
    window_days: int,
) -> list[Match]:
    since = request.since or datetime.now(UTC) - timedelta(days=window_days)
    vector = await _query_vector(embedder, request.q)
    candidates = await retriever.search(vector, since, request.collection, candidate_count)
    matches = await classify(request.q, candidates, ranker, batch_size)
    if request.threshold == "strong":
        matches = [m for m in matches if m.category == "strong"]
    # Newest first by ingestion time, id as the tie-break so equal timestamps
    # order the same way on every fetch.
    matches.sort(key=lambda m: (-m.item.ingested_at.timestamp(), m.item.id))
    return matches[: request.limit]


def render_rss(request: FeedRequest, matches: list[Match], base_url: str) -> bytes:
    ET.register_namespace("qd", NS)
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = f"rssnlweb: {request.q}"
    params = {"q": request.q}
    if request.collection:
        params["collection"] = request.collection
    if request.since:
        params["since"] = request.since.isoformat()
    ET.SubElement(channel, "link").text = f"{base_url}/feed.xml?{urlencode(params)}"
    ET.SubElement(channel, "description").text = (
        f"Items matching \"{request.q}\" at or above {request.threshold}"
    )
    ET.SubElement(channel, "lastBuildDate").text = format_datetime(datetime.now(UTC))
    for match in matches:
        item = match.item
        node = ET.SubElement(channel, "item")
        ET.SubElement(node, "title").text = item.title or item.url
        if item.url:
            ET.SubElement(node, "link").text = item.url
        ET.SubElement(node, "guid", isPermaLink="false").text = item.id
        ET.SubElement(node, "description").text = _description(match)
        if item.published_at:
            ET.SubElement(node, "pubDate").text = format_datetime(item.published_at)
        ET.SubElement(node, "source", url=item.url).text = item.source_title or item.source
        ET.SubElement(node, f"{{{NS}}}category").text = match.category
        ET.SubElement(node, f"{{{NS}}}ingestedAt").text = item.ingested_at.isoformat()
        ET.SubElement(node, f"{{{NS}}}source").text = item.source
    return ET.tostring(rss, encoding="utf-8", xml_declaration=True)


def _description(match: Match) -> str:
    why = f"[{match.category}] {match.why}".strip()
    body = match.item.content[:1000]
    return f"{why}\n\n{body}" if body else why
