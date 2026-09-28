from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Item:
    """One entry from one upstream feed. Immutable once stored."""

    id: str  # "<source>:<guid | link | hash>", unique across the corpus
    source: str  # source name from sources.yaml
    source_title: str
    url: str
    title: str
    content: str  # plain text, html stripped
    published_at: datetime | None  # source publication time, UTC
    published_raw: str  # as it appeared in the feed
    ingested_at: datetime  # first time we stored it, UTC; never changes
    collection: str = ""  # manifest section of the source; set at ingest

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.content}"


@dataclass(frozen=True)
class Candidate:
    item: Item
    lane: str  # "vector"
    score: float


@dataclass(frozen=True)
class Match:
    item: Item
    category: str  # "strong" | "relevant"
    why: str
