"""Write the in-memory snapshot of a collection (see qdrss/memory.py) from a local SQLite store.

    python scripts/export_snapshot.py tal OUTDIR            # then: gcloud storage cp OUTDIR/tal.* gs://BUCKET/PREFIX/

Reads the items and cached embeddings of the store at QDRSS_DATA_DIR (the one scripts/backfill_tal.py filled). Only
items that already have a vector are written. The app adds whatever Cosmos holds from after the snapshot, so it
needs rebuilding only to keep that catch-up small.
"""

import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from qdrss.config import load_config
from qdrss.memory import write_snapshot
from qdrss.store import Store


def main(collection: str, out: Path) -> None:
    load_dotenv()
    config = load_config()
    store = Store(config.db_path)
    items = [i for i in store.all_items() if i.collection == collection]
    vectors = store.embeddings(f"openrouter:{config.embedding_model}", [i.id for i in items])
    items = [i for i in items if i.id in vectors]
    write_snapshot(out, collection, items, np.vstack([vectors[i.id] for i in items]))
    print(f"{len(items)} items of {collection} -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
