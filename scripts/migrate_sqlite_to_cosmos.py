"""One-time load of a local SQLite corpus (items + cached vectors) into Cosmos DB.

    COSMOS_ENDPOINT=... COSMOS_KEY=... python scripts/migrate_sqlite_to_cosmos.py data/qdrss.sqlite

Idempotent: documents are upserted by id. Items without a vector for the
configured embedding model are written without one; `qdrss-ingest` embeds them.
"""

from __future__ import annotations

import asyncio
import sys
import time

from qdrss.config import load_config
from qdrss.cosmos import CosmosStore, _document
from qdrss.ingest import load_sources
from qdrss.store import Store


async def main(path: str) -> None:
    config = load_config()
    if not config.cosmos_endpoint:
        sys.exit("COSMOS_ENDPOINT / COSMOS_KEY must be set")
    collections = {s.name: s.collection for s in load_sources(config.sources_path)}
    model = config.embedding_model and f"openrouter:{config.embedding_model}"
    sqlite = Store(path)
    items = sqlite.all_items()
    vectors = sqlite.embeddings(model, [i.id for i in items])
    print(f"{len(items)} items, {len(vectors)} with {model} vectors")

    cosmos = CosmosStore(config.cosmos_endpoint, config.cosmos_key, config.cosmos_database)
    await cosmos.setup()
    sem = asyncio.Semaphore(16)
    done = 0
    started = time.time()

    async def put(item):
        nonlocal done
        from dataclasses import replace

        item = replace(item, collection=collections.get(item.source, item.collection or "default"))
        doc = _document(item, vectors.get(item.id), model if item.id in vectors else None)
        async with sem:
            await cosmos._items.upsert_item(doc)
        done += 1
        if done % 1000 == 0:
            print(f"{done}/{len(items)} in {time.time() - started:.0f}s")

    # Sources too, so the first ingest run sends conditional GETs.
    for src in await sqlite.all_sources():
        state = await sqlite.source_state(src["name"])
        await cosmos.remember_source(
            src["name"], src["url"], etag=state.get("etag"),
            last_modified=state.get("last_modified"), sha256=state.get("sha256"),
        )
    await asyncio.gather(*(put(item) for item in items))
    print(f"done: {done} documents, {await cosmos.count()} in Cosmos, {time.time() - started:.0f}s")
    await cosmos.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "data/qdrss.sqlite"))
