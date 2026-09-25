from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from .config import Config, load_config
from .feed import FeedRequest, evaluate, render_rss
from .feeds import parse_date
from .index import Index, build_index
from .ingest import load_sources, refresh
from .providers import Embeddings, HashEmbeddings, KeywordRanker, OpenRouter, Ranker
from .rank import RankingError
from .store import Store

logger = logging.getLogger("qdrss")
STATIC = Path(__file__).parent / "static"


class Services:
    def __init__(self, config: Config, embedder: Embeddings, ranker: Ranker):
        self.config = config
        self.embedder = embedder
        self.ranker = ranker
        self.store = Store(config.db_path)
        self.sources = load_sources(config.sources_path)
        self.collections = {s.name: s.collection for s in self.sources}
        self.index: Index = Index([], np.zeros((0, 1), dtype=np.float32))
        self.index_built_at: datetime | None = None
        self.last_refresh: dict = {}
        self._task: asyncio.Task | None = None

    async def refresh_once(self) -> None:
        outcomes = await refresh(
            self.sources, self.store, concurrency=self.config.fetch_concurrency
        )
        self.last_refresh = {
            "at": datetime.now(UTC).isoformat(),
            "new_items": sum(o.new_items for o in outcomes),
            "errors": {o.source.name: o.error for o in outcomes if o.error},
        }
        # Rebuild even when nothing changed: a failed previous build gets retried
        # for free, and the cost is bounded by the embedding cache.
        try:
            self.index = await build_index(self.store, self.embedder, self.collections)
            self.index_built_at = datetime.now(UTC)
        except Exception:
            logger.exception("index rebuild failed; serving the previous index")

    async def _loop(self) -> None:
        while True:
            try:
                await self.refresh_once()
            except Exception:
                logger.exception("refresh failed")
            await asyncio.sleep(self.config.refresh_minutes * 60)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self.store.close()


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
        if services.index_built_at is None:
            return PlainTextResponse("index not built yet", status_code=503)

        feed_request = FeedRequest(q, since, limit, threshold, collection)
        try:
            matches = await evaluate(
                feed_request,
                services.index,
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
        counts = services.store.counts_by_source()
        out: dict[str, dict] = {}
        for source in services.sources:
            entry = out.setdefault(source.collection, {"sources": 0, "items": 0})
            entry["sources"] += 1
            entry["items"] += counts.get(source.name, 0)
        return JSONResponse(out)

    async def health(_: Request) -> Response:
        return JSONResponse(
            {
                "items": services.store.count(),
                "index_built_at": (
                    services.index_built_at.isoformat() if services.index_built_at else None
                ),
                "last_refresh": services.last_refresh,
                "sources": services.store.all_sources(),
            }
        )

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette):
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
