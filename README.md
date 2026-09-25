# QD RSS — Query Defined RSS

A natural-language query is a feed. The server keeps a corpus of upstream RSS/Atom
items and, when a feed is fetched, returns the newest items that match the query.

    GET /feed.xml?q=<query>[&since=<date>][&limit=N][&threshold=relevant|strong]

The server keeps state about the feeds (items, fetch validators, embeddings) and
nothing about users: no saved queries, no subscriber tokens, no watermarks. The URL
is the whole definition of a feed.

## Run

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env            # OPENROUTER_API_KEY for real embeddings and ranking
qdrss
```

Open <http://127.0.0.1:8000/>: type a query, get the feed URL and a preview of what it
returns now. The URL encodes the query; there is nothing to save on the server. `/health` shows item
count, index age, and per-source fetch status. Without `OPENROUTER_API_KEY` the
server uses hash embeddings and a keyword ranker: enough to see the plumbing work,
not enough to judge relevance.

## How a fetch works

1. `since` (or, when absent, now minus `QDRSS_WINDOW_DAYS`) masks the index to items
   **ingested** after that time (and to one collection, if `collection=` is given).
   The mask is applied before top-K, so the candidates are the best *new* items,
   not the new items among the best overall.
2. Vector retrieval over the masked items yields up to `QDRSS_CANDIDATE_COUNT` (40)
   candidates. There is no lexical lane.
3. The ranking model classifies every candidate as `strong`, `relevant`, or
   excluded. No per-batch quota. If the model call fails the response is 503, not an
   empty feed.
4. Survivors at or above `threshold` are ordered newest-first by ingestion time (item
   id as tie-break) and capped at `limit`.

## Semantics to know

- **`since` is on ingestion time, not publication time.** Each item carries
  `<qd:ingestedAt>`; a client that passes the newest one back as `since` gets exactly
  what arrived after it, including late-published items. The first import of a
  corpus stamps every existing item "now", so a fresh server shows its whole back
  catalog as new for one window.
- **A capped response is not an exact delta.** The pool is bounded (40 candidates)
  and the output is capped, so a client advancing `since` past a full page may skip
  matches. Readers that fetch the fixed URL get the trailing window each time and
  dedupe on GUID; that is the intended mode.
- **Items are append-only.** First stored version wins; upstream edits and deletions
  are ignored, and items that roll off a feed stay in the corpus.
- **Identity** is `<source>:<guid>`, else the link, else a hash of title + raw date.
  GUIDs are namespaced by source because upstream ids are only unique per feed.
- `pubDate` is the source publication time. `<qd:category>` and `<qd:source>` carry
  the verdict and the source name.

## Sources

`sources.yaml` lists the upstream feeds. They are fetched at startup and every
`QDRSS_REFRESH_MINUTES` with ETag/Last-Modified; servers that ignore those are caught
by a body hash. A malformed or failing feed keeps its previous items and validators
and does not block the others. On a source's first fetch, ingest follows RFC 5005 `rel="next"` archive pages so a
paginated feed's whole history lands in the store; later fetches read page one only.
A feed the publisher truncates (some NYT and TWiT shows) yields only what it carries.
The index is rebuilt after every refresh from the SQLite store; only items without a
cached embedding are embedded.

## Tests

```bash
pytest
```

Fake providers throughout; the conditional-GET paths use `httpx.MockTransport`.
