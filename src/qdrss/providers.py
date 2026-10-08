"""Embeddings and the ranking model. Fakes for offline runs and tests."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import deque
from typing import Any, Protocol

import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class Embeddings(Protocol):
    model: str  # cache key

    async def embed(self, texts: list[str]) -> np.ndarray: ...


class Ranker(Protocol):
    async def structured(self, instruction: str, payload: dict[str, Any]) -> dict[str, Any]: ...


class HashEmbeddings:
    """Deterministic bag-of-hashed-tokens. Offline only; not semantic."""

    model = "hash-v1-384"
    dimensions = 384

    async def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                number = int.from_bytes(digest, "little")
                out[row, number % self.dimensions] += 1 if number & 1 else -1
            norm = float(np.linalg.norm(out[row]))
            if norm:
                out[row] /= norm
        return out


class KeywordRanker:
    """Offline stand-in: strong if every query word appears, relevant if half do."""

    async def structured(self, instruction: str, payload: dict[str, Any]) -> dict[str, Any]:
        words = set(re.findall(r"[a-z0-9]+", payload["q"].lower()))
        results = []
        for record in payload["rs"]:
            text = f"{record['title']} {record['text']}".lower()
            hits = sum(1 for w in words if w in text)
            if not words or hits == 0:
                m = "exclude"
            elif hits == len(words):
                m = "strong"
            elif hits * 2 >= len(words):
                m = "relevant"
            else:
                m = "exclude"
            results.append({"i": record["i"], "m": m, "why": f"{hits} of {len(words)} terms"})
        return {"results": results}


class ProviderHealth:
    """Which OpenRouter providers have been failing lately, remembered for this process.

    A call counts as a failure when the provider returned nothing or unparseable JSON. A provider with at least
    `min_failures` failures in the last `window` seconds, making up at least `max_share` of its calls in that time,
    is avoided: it is added to the ignore list of new calls. The record ages out, so a provider is tried again once
    its failures are old enough, and one that is still bad is dropped again after a few more.
    """

    def __init__(self, window: float = 3600, min_failures: int = 3, max_share: float = 0.3, clock=time.time):
        self.window, self.min_failures, self.max_share, self.clock = window, min_failures, max_share, clock
        self.events: dict[str, deque[tuple[float, bool]]] = {}
        self.last_problem: dict[str, str] = {}

    def record(self, provider: str | None, ok: bool, *, problem: str = "") -> None:
        if provider:
            self.events.setdefault(provider, deque()).append((self.clock(), ok))
            if not ok:
                self.last_problem[provider] = problem[:200]

    def _recent(self, provider: str) -> list[bool]:
        events = self.events.get(provider, deque())
        while events and events[0][0] < self.clock() - self.window:
            events.popleft()
        return [ok for _, ok in events]

    def avoid(self) -> list[str]:
        out = []
        for provider in list(self.events):
            recent = self._recent(provider)
            failures = recent.count(False)
            if failures >= self.min_failures and failures / len(recent) >= self.max_share:
                out.append(provider)
        return sorted(out)

    def report(self) -> dict:
        rows = {}
        for provider in sorted(self.events):
            recent = self._recent(provider)
            if recent:
                rows[provider] = {"calls": len(recent), "failures": recent.count(False),
                                  "avoided": provider in self.avoid(), "last_problem": self.last_problem.get(provider, "")}
        return rows


class OpenRouter:
    def __init__(
        self,
        key: str,
        base_url: str,
        ranking_model: str,
        embedding_model: str,
        provider_sort: str | None = "throughput",
        timeout: float = 60.0,
    ):
        self.client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=1)
        self.ranking_model = ranking_model
        self.provider_sort = provider_sort
        self.model = f"openrouter:{embedding_model}"
        self.embedding_model = embedding_model
        self.health = ProviderHealth()

    async def embed(self, texts: list[str]) -> np.ndarray:
        response = await self.client.embeddings.create(model=self.embedding_model, input=texts)
        matrix = np.asarray([d.embedding for d in response.data], dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        return matrix

    async def _ranking_call(self, instruction: str, payload: dict[str, Any], ignore: list[str]):
        provider: dict[str, Any] = {}
        if self.provider_sort:
            provider["sort"] = self.provider_sort
        avoided = [p for p in self.health.avoid() if p not in ignore]
        if avoided:
            provider["ignore"] = avoided
        if ignore:
            # A retry: skip the provider that failed, and take only providers that support every
            # parameter sent (json_object output), so the second try is a more capable one.
            provider["ignore"] = sorted({*ignore, *avoided})
            provider["require_parameters"] = True
        return await self.client.chat.completions.create(
            model=self.ranking_model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            # Reasoning models spend max_tokens on thinking first, so effort is
            # low; the cap bounds the damage when a model degenerates into
            # thousands of lines of broken JSON (seen with gpt-oss-20b).
            max_tokens=1500,
            extra_body={"reasoning": {"effort": "low"}, **({"provider": provider} if provider else {})},
        )

    @staticmethod
    def _parse(response) -> tuple[Any, str]:
        """(decoded JSON, "") or (None, what is wrong with the reply)."""
        choice = response.choices[0]
        if not choice.message.content:
            return None, f"an empty response (finish={choice.finish_reason})"
        try:
            return json.loads(choice.message.content), ""
        except json.JSONDecodeError:
            head = choice.message.content[:160].replace("\n", "\\n")
            return None, f"malformed JSON ({len(choice.message.content)} chars, finish={choice.finish_reason}): {head!r}"

    async def structured(self, instruction: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._ranking_call(instruction, payload, [])
        result, problem = self._parse(response)
        self.health.record(getattr(response, "provider", None), not problem, problem=problem)
        if problem:
            # A provider can end a call with no content (finish_reason "error", seen with Groq's
            # gpt-oss-20b) or degenerate into junk until max_tokens (finish "length"). Temperature is 0,
            # so the same provider fails the same way on every retry and the query 503s for good.
            # Try again with that provider excluded.
            failed = getattr(response, "provider", None)
            logger.warning("ranking call failed: %s (provider=%s)%s", problem, failed,
                           "; retrying without it" if failed else "")
            if failed:
                response = await self._ranking_call(instruction, payload, [failed])
                result, problem = self._parse(response)
                self.health.record(getattr(response, "provider", None), not problem, problem=problem)
        if problem:
            raise (RuntimeError if problem.startswith("an empty") else ValueError)(f"ranking model returned {problem}")
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
            result = result[0]
        if not isinstance(result, dict):
            raise TypeError("ranking model returned JSON of an unsupported shape")
        return result
