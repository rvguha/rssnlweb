from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    data_dir: Path
    sources_path: Path
    host: str
    port: int
    refresh_minutes: int
    window_days: int
    candidate_count: int
    default_limit: int
    max_limit: int
    batch_size: int
    fetch_concurrency: int
    openrouter_key: str
    openrouter_base_url: str
    ranking_model: str
    embedding_model: str
    provider_sort: str

    @property
    def db_path(self) -> Path:
        return self.data_dir / "qdrss.sqlite"


def load_config() -> Config:
    load_dotenv()
    env = os.environ
    return Config(
        data_dir=Path(env.get("QDRSS_DATA_DIR", "data")),
        sources_path=Path(env.get("QDRSS_SOURCES", "sources.yaml")),
        host=env.get("QDRSS_HOST", "127.0.0.1"),
        port=int(env.get("QDRSS_PORT", "8000")),
        refresh_minutes=int(env.get("QDRSS_REFRESH_MINUTES", "60")),
        window_days=int(env.get("QDRSS_WINDOW_DAYS", "7")),
        candidate_count=int(env.get("QDRSS_CANDIDATE_COUNT", "40")),
        default_limit=int(env.get("QDRSS_DEFAULT_LIMIT", "20")),
        max_limit=int(env.get("QDRSS_MAX_LIMIT", "100")),
        batch_size=int(env.get("QDRSS_RANKING_BATCH_SIZE", "5")),
        fetch_concurrency=int(env.get("QDRSS_FETCH_CONCURRENCY", "6")),
        openrouter_key=env.get("OPENROUTER_API_KEY", ""),
        openrouter_base_url=env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        ranking_model=env.get("OPENROUTER_RANKING_MODEL", "openai/gpt-oss-20b"),
        embedding_model=env.get("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        provider_sort=env.get("OPENROUTER_RANKING_PROVIDER_SORT", "throughput"),
    )
