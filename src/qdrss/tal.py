"""This American Life, split into sections with an offset into the audio.

A TAL episode is an hour of radio in a few acts. The feed gives one item per episode, which cannot say *where*
in the hour something is said. The site publishes a transcript for every episode (thisamericanlife.org/<n>/transcript):
a block per act, and every paragraph carries its start time in seconds. This module turns an episode into
passages of about two minutes, each an Item of its own with the act it belongs to and the second it starts at,
so a query returns "Act Two, 33:45" instead of "episode 899".

The episode page (thisamericanlife.org/<n>/<slug>) adds what the transcript lacks: the air date and a public MP3
that honours byte ranges, so a result can link to the audio at `#t=<seconds>`.

The transcript's act labels are the transcribers' and are not always right (in #899 the block labelled Prologue holds
the introduction and Part One). The paragraph times are right, so the offset of a passage is; only its act name can
be off.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from .models import Item

logger = logging.getLogger(__name__)

SITE = "https://www.thisamericanlife.org"
WINDOW_WORDS = 280  # about two minutes of speech; fits what the ranker sees of an item
MIN_TAIL_WORDS = 80  # a shorter last passage joins the one before it
_ACT = re.compile(r'<div class="act" id="([^"]*)">\s*<h3>(.*?)</h3>(.*?)(?=<div class="act" id=|</main>|<footer|\Z)', re.S)
_TOKEN = re.compile(r"<h4>(.*?)</h4>|<p[^>]*data-timestamp=\"([\d.]+)\"[^>]*>(.*?)</p>", re.S)


@dataclass(frozen=True)
class Paragraph:
    start: float
    speaker: str
    text: str


@dataclass(frozen=True)
class Act:
    id: str
    label: str
    paragraphs: tuple[Paragraph, ...]


@dataclass(frozen=True)
class Passage:
    act_id: str
    act: str
    start: float
    end: float
    text: str


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def act_label(raw: str) -> str:
    """"Prologue: Prologue" -> "Prologue"; "Act One: The Intern" is kept."""
    head, sep, tail = raw.partition(":")
    return head.strip() if sep and head.strip().casefold() == tail.strip().casefold() else raw.strip()


def parse_transcript(page: str) -> list[Act]:
    acts = []
    for act_id, heading, body in _ACT.findall(page):
        speaker, paragraphs = "", []
        for name, stamp, para in _TOKEN.findall(body):
            if name:
                speaker = _text(name)
            elif _text(para):
                paragraphs.append(Paragraph(float(stamp), speaker, _text(para)))
        if paragraphs:
            acts.append(Act(act_id, act_label(_text(heading)), tuple(paragraphs)))
    return acts


def passages(acts: list[Act], window_words: int = WINDOW_WORDS) -> list[Passage]:
    """About `window_words` of consecutive paragraphs, never across acts. A passage starts at its first paragraph's time
    and ends where the next begins (or at its own last paragraph); the speaker is named when it changes."""
    out: list[Passage] = []
    for act in acts:
        chunks: list[list[Paragraph]] = [[]]
        words = 0
        for p in act.paragraphs:
            chunks[-1].append(p)
            words += len(p.text.split())
            if words >= window_words:
                chunks.append([])
                words = 0
        if len(chunks) > 1 and not chunks[-1]:
            chunks.pop()
        if len(chunks) > 1 and sum(len(p.text.split()) for p in chunks[-1]) < MIN_TAIL_WORDS:
            chunks[-2].extend(chunks.pop())
        for i, chunk in enumerate(chunks):
            lines, last = [], ""
            for p in chunk:
                lines.append(f"{p.speaker}: {p.text}" if p.speaker and p.speaker != last else p.text)
                last = p.speaker
            end = chunks[i + 1][0].start if i + 1 < len(chunks) else chunk[-1].start
            out.append(Passage(act.id, act.label, chunk[0].start, end, " ".join(lines)))
    return out


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


@dataclass(frozen=True)
class Episode:
    number: int
    title: str
    page: str  # the episode page URL
    transcript: str
    aired: datetime | None
    audio: str


def _first(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.S)
    return m.group(1) if m else ""


def slugify(title: str) -> str:
    """The address TAL derives from an episode title: "I Want What I Want" -> "i-want-what-i-want"."""
    words = re.sub(r"[^a-z0-9 ]+", "", title.casefold().replace("’", "").replace("'", "")).split()
    return "-".join(words)


def episode_title(transcript_page: str, number: int) -> str:
    return _text(_first(r'property="og:title"[^>]+content="([^"]+)"', transcript_page)).removesuffix(" - This American Life") or f"{number}"


def episode_info(number: int, transcript_page: str, episode_page: str = "", audio: str = "") -> Episode:
    """`audio` is what the caller found (the feed's, the page's, or a checked guess); the page's link is the fallback.
    The air date is the episode page's own date field; failing that the page's publish time, then the transcript's."""
    title = episode_title(transcript_page, number)
    slug = _first(r'href="(/%d/(?!transcript)[^"#?]+)"' % number, transcript_page)
    shown = _first(r'class="[^"]*field-name-field-radio-air-date[^"]*"[^>]*>.*?([A-Z][a-z]{2,8}\.? \d{1,2}, \d{4})', episode_page) or _first(
        r'class="[^"]*(?:air|date)[^"]*"[^>]*>\s*([A-Z][a-z]{2,8}\.? \d{1,2}, \d{4})\s*<', episode_page
    )
    aired = None
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            aired = datetime.strptime(shown.replace(".", ""), fmt).replace(tzinfo=UTC)
            break
        except ValueError:
            continue
    for page in (episode_page, transcript_page):
        stamp = _first(r'<meta[^>]+property="article:published_time"[^>]+content="([^"]+)"', page)
        if aired is None and stamp:
            try:
                aired = datetime.fromisoformat(stamp).astimezone(UTC)
            except ValueError:
                pass
    audio = audio or _first(r'href="(https://www\.thisamericanlife\.org/sites/default/files/audio/[^"]+\.mp3)"', episode_page)
    return Episode(number, title, f"{SITE}{slug}" if slug else f"{SITE}/{number}/transcript",
                   f"{SITE}/{number}/transcript", aired, audio)


def section_items(episode: Episode, transcript_page: str, *, source: str, collection: str, ingested_at: datetime) -> list[Item]:
    acts = parse_transcript(transcript_page)
    items = []
    counts: dict[str, int] = {}
    for p in passages(acts):
        n = counts[p.act_id] = counts.get(p.act_id, 0) + 1
        extra = {"episode": episode.number, "actId": p.act_id, "act": p.act, "start": round(p.start, 1), "end": round(p.end, 1),
                 "transcript": f"{episode.transcript}#{p.act_id}"}
        if episode.audio:
            extra["audio"] = f"{episode.audio}#t={int(p.start)}"
        items.append(
            Item(
                id=f"{source}:{episode.number}:{p.act_id}:{n:03d}",
                source=source,
                source_title="This American Life",
                url=episode.page,
                title=f"{episode.title} — {p.act}" if p.act else episode.title,
                content=p.text,
                published_at=episode.aired,
                published_raw=episode.aired.strftime("%a, %d %b %Y") if episode.aired else "",
                ingested_at=ingested_at,
                collection=collection,
                extra=json.dumps(extra, separators=(",", ":")),
            )
        )
    return items


async def _get(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    """A GET that treats throttling and server errors as errors (raised), not as an empty answer."""
    response = await client.get(url, **kwargs)
    if response.status_code == 429 or response.status_code >= 500:
        raise httpx.HTTPStatusError(f"HTTP {response.status_code}", request=response.request, response=response)
    return response


async def fetch_pages(number: int, client: httpx.AsyncClient, *, feed_audio: str = "") -> tuple[str, str, str] | None:
    """(transcript page, episode page, audio URL) for one episode, or None when the site has no timed transcript.
    The episode page is found by the transcript's link to it, else by the address its title implies ("/824/family-meeting"
    redirects to "/family-meeting"). The audio is the feed's, else the page's, else the site's standard address when a
    one-byte range request shows it serves audio (the recent "clean" and "archive" locations). Throttling and server errors raise; a missing page is just empty."""
    response = await _get(client, f"{SITE}/{number}/transcript")
    if response.status_code != 200 or 'data-timestamp="' not in response.text:
        return None
    transcript = response.text
    slug = _first(r'href="(/%d/(?!transcript)[^"#?]+)"' % number, transcript)
    guesses = [SITE + slug] if slug else []
    implied = slugify(episode_title(transcript, number).split(":", 1)[-1])
    guesses += [f"{SITE}/{number}/{implied}", f"{SITE}/{implied}"]      # recent episodes live at /<slug>, older at /<n>/<slug>
    episode_page = ""
    for url in guesses:
        page = await _get(client, url)
        if page.status_code == 200 and "audio" in page.text:
            episode_page = page.text
            break
    audio = feed_audio or _first(r'href="(https://www\.thisamericanlife\.org/sites/default/files/audio/[^"]+\.mp3)"', episode_page)
    for where in ("clean", "archive") if not audio else ():
        guess = f"{SITE}/sites/default/files/audio/upload/{'schedule/clean' if where == 'clean' else 'archive'}/{number}.mp3"
        probe = await _get(client, guess, headers={"Range": "bytes=0-0"})
        if probe.status_code in (200, 206) and "audio" in probe.headers.get("content-type", ""):
            audio = guess
            break
    return transcript, episode_page, audio


def feed_episodes(body: bytes) -> dict[int, str]:
    """Episode number -> audio URL for every item in the TAL feed."""
    import xml.etree.ElementTree as ET

    out = {}
    for item in ET.fromstring(body).iter("item"):
        number = next((c.text for c in item if c.tag.endswith("}episode") and (c.text or "").isdigit()), None)
        title = next((c.text for c in item if c.tag == "title"), "") or ""
        number = number or _first(r"^\s*(\d+):", title)
        enclosure = next((c.get("url", "") for c in item if c.tag == "enclosure"), "")
        if number and str(number).isdigit():
            out[int(number)] = enclosure
    return out


async def expand(feed_body: bytes, source: str, collection: str, client: httpx.AsyncClient, limit: asyncio.Semaphore) -> list[Item]:
    """The feed's episodes, as sections. An episode whose transcript is not up yet is skipped; the next fetch tries again
    (ids are fixed, so a section already stored is never duplicated)."""
    now = datetime.now(UTC)

    async def one(number: int, audio: str) -> list[Item]:
        async with limit:
            try:
                got = await fetch_pages(number, client, feed_audio=audio)
            except httpx.HTTPError as exc:
                logger.warning("%s: episode %d: %s", source, number, exc)
                return []
        if got is None:
            logger.info("%s: episode %d has no transcript yet", source, number)
            return []
        transcript, episode_page, found = got
        episode = episode_info(number, transcript, episode_page, found)
        return section_items(episode, transcript, source=source, collection=collection, ingested_at=now)

    batches = await asyncio.gather(*(one(n, a) for n, a in feed_episodes(feed_body).items()))
    return [item for batch in batches for item in batch]
