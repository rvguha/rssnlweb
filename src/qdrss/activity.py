"""What the service has been doing: feed requests answered, when the last one was, how long they take and how they ended.

The counters live in this process, so they restart at zero with the service and each instance counts its own.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from datetime import UTC, datetime


def iso(epoch: float | None) -> str | None:
    return None if epoch is None else datetime.fromtimestamp(epoch, UTC).isoformat(timespec="seconds")


def percentile(values, p: float):
    values = sorted(values)
    return None if not values else round(values[min(len(values) - 1, int(round(p * (len(values) - 1))))], 3)


class Activity:
    def __init__(self, started: float | None = None, window: int = 500):
        self.started = started if started is not None else time.time()
        self.by_status: Counter = Counter()
        self.latencies: deque = deque(maxlen=window)
        self.last_at: float | None = None
        self.last_error: str | None = None
        self.last_error_at: float | None = None
        self._lock = threading.Lock()

    def record(self, status: int, seconds: float, error: str | None = None, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        with self._lock:
            self.by_status[str(status)] += 1
            self.last_at = now
            self.latencies.append(float(seconds))
            if status >= 500:
                self.last_error, self.last_error_at = (error or f"HTTP {status}")[:200], now

    def snapshot(self, now: float | None = None) -> dict:
        now = now if now is not None else time.time()
        with self._lock:
            seen = list(self.latencies)
            total = sum(self.by_status.values())
            return {
                "total": total,
                "by_status": dict(sorted(self.by_status.items())),
                "server_errors": sum(n for code, n in self.by_status.items() if code.startswith("5")),
                "last_at": iso(self.last_at),
                "seconds_since_last": None if self.last_at is None else round(now - self.last_at, 1),
                "latency_seconds": {"p50": percentile(seen, 0.5), "p95": percentile(seen, 0.95), "sampled": len(seen)},
                "last_error": self.last_error, "last_error_at": iso(self.last_error_at),
            }
