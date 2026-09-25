from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from qdrss.models import Item
from qdrss.store import Store

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)

RSS = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
<channel><title>Feed A</title>
<item><title>Rust 2.0 released</title><link>https://a.test/rust</link>
  <guid isPermaLink="false">a-1</guid><pubDate>Mon, 21 Sep 2026 10:00:00 GMT</pubDate>
  <description>&lt;p&gt;The Rust language ships version 2.0 &amp; a new borrow checker.&lt;/p&gt;</description></item>
<item><title>Untitled &lt;b&gt;bold&lt;/b&gt;</title><link>https://a.test/no-guid</link>
  <content:encoded>Encoded body wins</content:encoded><description>ignored</description></item>
<item><title>No guid no link</title><pubDate>Tue, 22 Sep 2026 09:00:00 +0000</pubDate>
  <description>Only a hash identifies this one</description></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Feed B</title>
<entry><title>Python 4 plans</title><id>urn:b:1</id>
  <link rel="enclosure" href="https://b.test/audio.mp3"/>
  <link rel="alternate" href="https://b.test/python4"/>
  <published>2026-09-22T08:30:00Z</published><summary>Guido on Python 4 and typing.</summary></entry>
</feed>"""


def make_item(
    id: str, title: str, content: str = "", *, ingested: datetime = NOW, source: str = "s"
) -> Item:
    return Item(
        id=f"{source}:{id}",
        source=source,
        source_title="Source",
        url=f"https://{source}.test/{id}",
        title=title,
        content=content,
        published_at=ingested - timedelta(hours=1),
        published_raw="",
        ingested_at=ingested,
    )


@pytest.fixture
def store() -> Store:
    return Store(":memory:")
