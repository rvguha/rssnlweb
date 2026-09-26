from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from .config import Config, load_config
from .feed import FeedRequest, Retriever, evaluate, render_rss
from .feeds import parse_date
from .index import Index, build_index
from .ingest import load_sources, refresh
from .providers import Embeddings, HashEmbeddings, KeywordRanker, OpenRouter, Ranker
from .rank import RankingError

logger = logging.getLogger("qdrss")
STATIC = Path(__file__).parent / "static"


class Services:
    """Two storage modes behind one interface.

    Cosmos (COSMOS_ENDPOINT set): items and vectors live in Cosmos DB, retrieval
    is a Cosmos vector query, and the process holds nothing. SQLite (default):
    items live in a local file and an in-memory matrix is rebuilt after every
    refresh. Tests and offline runs use SQLite.
    """

    def __init__(self, config: Config, embedder: Embeddings, ranker: Ranker):
        self.config = config
        self.embedder = embedder
        self.ranker = ranker
        self.sources = load_sources(config.sources_path)
        self.collections = {s.name: s.collection for s in self.sources}
        self.cosmos = bool(config.cosmos_endpoint)
        if self.cosmos:
            from .cosmos import CosmosStore

            self.store = CosmosStore(config.cosmos_endpoint, config.cosmos_key, config.cosmos_database)
        else:
            from .store import Store

            self.store = Store(config.db_path)
        self.index: Index = Index([], np.zeros((0, 1), dtype=np.float32))
        self.ready_at: datetime | None = None
        self.last_refresh: dict = {}
        self._task: asyncio.Task | None = None
        self._counts_cache: tuple[float, dict] | None = None

    @property
    def retriever(self) -> Retriever:
        return self.store if self.cosmos else self.index  # type: ignore[return-value]

    async def setup(self) -> None:
        await self.store.setup()
        if self.cosmos:
            self.ready_at = datetime.now(UTC)

    async def refresh_once(self) -> None:
        outcomes = await refresh(
            self.sources, self.store, concurrency=self.config.fetch_concurrency
        )
        self.last_refresh = {
            "at": datetime.now(UTC).isoformat(),
            "new_items": sum(o.new_items for o in outcomes),
            "errors": {o.source.name: o.error for o in outcomes if o.error},
        }
        try:
            if self.cosmos:
                await self.embed_missing()
            else:
                # Rebuild even when nothing changed: a failed previous build gets
                # retried for free, and the cost is bounded by the embedding cache.
                self.index = await build_index(self.store, self.embedder, self.collections)
            self.ready_at = datetime.now(UTC)
        except Exception:
            logger.exception("indexing failed; serving the previous state")
        self._counts_cache = None

    async def embed_missing(self) -> int:
        """Cosmos mode: embed items that arrived without a vector, in batches."""
        total = 0
        while True:
            items = await self.store.items_without_embedding(100)
            if not items:
                return total
            vectors = await self.embedder.embed([item.text[:8000] for item in items])
            await self.store.save_embeddings(
                self.embedder.model, {item.id: vectors[i] for i, item in enumerate(items)}
            )
            total += len(items)
            logger.info("embedded %d items (%d so far)", len(items), total)

    async def collection_counts(self) -> dict[str, dict]:
        if self._counts_cache and time.monotonic() - self._counts_cache[0] < 60:
            return self._counts_cache[1]
        out: dict[str, dict] = {}
        for source in self.sources:
            out.setdefault(source.collection, {"sources": 0, "items": 0})["sources"] += 1
        if self.cosmos:
            for name, n in (await self.store.counts_by_collection(list(out))).items():
                out.setdefault(name, {"sources": 0, "items": 0})["items"] = n
        else:
            counts = self.store.counts_by_source()
            for source in self.sources:
                out[source.collection]["items"] += counts.get(source.name, 0)
        self._counts_cache = (time.monotonic(), out)
        return out

    async def _loop(self) -> None:
        while True:
            try:
                await self.refresh_once()
            except Exception:
                logger.exception("refresh failed")
            await asyncio.sleep(self.config.refresh_minutes * 60)

    def start(self) -> None:
        if self.config.ingest_in_app or not self.cosmos:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self.store.close()


def build_services(config: Config | None = None) -> Services:
    config = config or load_config()
    if config.openrouter_key:
        provider = OpenRouter(
            config.openrouter_key,
            config.openrouter_base_url,
            config.ranking_model,
            config.embedding_model,
            provider_sort=config.provider_sort or None,
        )
        return Services(config, provider, provider)
    logger.warning("OPENROUTER_API_KEY not set: using hash embeddings and keyword ranking")
    return Services(config, HashEmbeddings(), KeywordRanker())


def create_app(services: Services) -> Starlette:
    async def feed(request: Request) -> Response:
        params = request.query_params
        q = params.get("q", "").strip()
        if not q:
            return PlainTextResponse("q is required", status_code=400)
        since = None
        if raw := params.get("since", "").strip():
            since = parse_date(raw)
            if since is None:
                return PlainTextResponse("since must be RFC 2822 or ISO 8601", status_code=400)
        try:
            limit = int(params.get("limit", services.config.default_limit))
        except ValueError:
            return PlainTextResponse("limit must be an integer", status_code=400)
        limit = max(1, min(limit, services.config.max_limit))
        threshold = params.get("threshold", "relevant")
        if threshold not in ("relevant", "strong"):
            return PlainTextResponse("threshold must be relevant or strong", status_code=400)
        collection = params.get("collection", "").strip() or None
        if collection and collection not in services.collections.values():
            return PlainTextResponse(f"unknown collection {collection!r}", status_code=400)
        if services.ready_at is None:
            return PlainTextResponse("index not built yet", status_code=503)

        feed_request = FeedRequest(q, since, limit, threshold, collection)
        try:
            matches = await evaluate(
                feed_request,
                services.retriever,
                services.embedder,
                services.ranker,
                candidate_count=services.config.candidate_count,
                batch_size=services.config.batch_size,
                window_days=services.config.window_days,
            )
        except RankingError as exc:
            logger.error("ranking failed: %s", exc)
            return PlainTextResponse(f"ranking unavailable: {exc}", status_code=503)
        base = str(request.base_url).rstrip("/")
        body = render_rss(feed_request, matches, base)
        return Response(body, media_type="application/rss+xml; charset=utf-8")

    async def home(_: Request) -> Response:
        return FileResponse(STATIC / "index.html")

    async def collections(_: Request) -> Response:
        return JSONResponse(await services.collection_counts())

    async def health(_: Request) -> Response:
        return JSONResponse(
            {
                "store": "cosmos" if services.cosmos else "sqlite",
                "items": await services.store.count(),
                "index_built_at": services.ready_at.isoformat() if services.ready_at else None,
                "last_refresh": services.last_refresh,
                "sources": await services.store.all_sources(),
            }
        )

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette):
        await services.setup()
        services.start()
        try:
            yield
        finally:
            await services.stop()

    return Starlette(
        routes=[
            Route("/", home),
            Route("/feed.xml", feed),
            Route("/collections", collections),
            Route("/health", health),
        ],
        lifespan=lifespan,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    services = build_services()
    uvicorn.run(create_app(services), host=services.config.host, port=services.config.port)


def ingest_main() -> None:
    """Fetch every source once, store new items, embed them, exit.

    Runs anywhere with the same .env as the web app; with Cosmos this is how the
    corpus is updated independently of serving.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    async def run() -> None:
        services = build_services()
        await services.setup()
        try:
            await services.refresh_once()
            r = services.last_refresh
            print(f"new items: {r['new_items']}; errors: {len(r['errors'])}")
            for name, err in r["errors"].items():
                print(f"  {name}: {err}")
        finally:
            await services.store.close()

    asyncio.run(run())
