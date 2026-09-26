# rssnlweb — Query Defined RSS

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
cp .env.example .env            # then add your key, see below
qdrss
```

Open <http://127.0.0.1:8000/>: type a query, get the feed URL and a preview of what it
returns now. The URL encodes the query; there is nothing to save on the server.
`/health` shows item count, index age, and per-source fetch status.

## Keys and models

All model calls go through [OpenRouter](https://openrouter.ai), one key for both
embeddings and ranking. Put it in `.env` (git-ignored) or in the environment:

```bash
OPENROUTER_API_KEY=sk-or-...            # https://openrouter.ai/settings/keys
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_RANKING_MODEL=openai/gpt-oss-20b
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENROUTER_RANKING_PROVIDER_SORT=throughput
```

Without a key the server falls back to hash embeddings and a keyword ranker: the
plumbing works, the relevance doesn't. Don't judge results in that mode.

**Ranking model.** The ranker classifies each candidate strong / relevant / exclude
and returns JSON. Any OpenRouter chat model that honours `response_format:
json_object` works. Open-weight options, cheapest first:

| Model | Notes |
|---|---|
| `openai/gpt-oss-20b` | default; measured ~1 s per batch of 5 with `provider.sort=throughput` (10 s without) |
| `openai/gpt-oss-120b` | same family, better judgement, ~3x the price |
| `google/gemma-3-27b-it` | fast, no reasoning tokens |
| `meta-llama/llama-3.3-70b-instruct` | solid at structured output |
| `mistralai/mistral-small-3.2-24b-instruct` | cheap, fast |
| `deepseek/deepseek-v3.2` | strongest of the open models here; slower |

Only `gpt-oss-20b` has been measured on this workload. Two things matter for reasoning
models like gpt-oss: the request sets `reasoning.effort=low` and a generous
`max_tokens`, because a tight cap makes them spend the budget thinking and return
empty content (which this server reports as a 503, not an empty feed).

**Embedding model.** `openai/text-embedding-3-small` (1536 dims) is what the shipped
corpus was embedded with. Embeddings are cached per item per model, so changing
`OPENROUTER_EMBEDDING_MODEL` re-embeds every item on the next refresh; at 57k items
that is about $0.15 and 15 minutes with text-embedding-3-small, and the old vectors
stay in the store keyed by their model name. Any model OpenRouter's `/embeddings`
endpoint serves can be named here; open-weight embedding models come and go on
OpenRouter, so check their catalogue before switching.

**Cost.** A feed fetch is one embedding call (cached per query) plus
`candidate_count / batch_size` ranking calls (8 by default). With gpt-oss-20b that is
about 0.2 cents per fetch, so **1,000 fetches cost about $2**. Every fetch pays that:
the rendered feed is not cached, so a reader polling a URL hourly costs ~$1.40 a month
on its own, and 100 readers on the same URL cost 100x. Only the query's embedding is
cached. The corpus embedding is a one-time cost per model.

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

## Storage: Cosmos DB or SQLite

Two modes behind one interface, chosen by whether `COSMOS_ENDPOINT` is set.

- **Cosmos DB for NoSQL** (production): one `items` container partitioned by
  collection holds each episode with its embedding; the feed's nearest-neighbour
  query runs in Cosmos with the date and collection filters, over a DiskANN index.
  The web process holds nothing. The free tier (1000 RU/s, 25 GB) covers tens of
  thousands of episodes. `qdrss-ingest` fetches feeds, stores new items and embeds
  them; run it on a schedule anywhere with the same `.env`, or leave
  `QDRSS_INGEST_IN_APP=true` and the web app runs it hourly itself.
  `scripts/migrate_sqlite_to_cosmos.py` loads an existing SQLite corpus, vectors
  included, without re-embedding.
- **SQLite** (default, tests, offline): items and cached vectors in one file, an
  in-memory matrix rebuilt after each refresh.

## Sources

`sources.yaml` lists the upstream feeds, grouped into collections. They are fetched at
startup and every `QDRSS_REFRESH_MINUTES` (360 in production, i.e. four times a day)
with ETag/Last-Modified; new episodes are inserted by id, so nothing is ever
duplicated; servers that ignore those are caught
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
