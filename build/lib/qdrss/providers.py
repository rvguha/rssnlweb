"""Embeddings and the ranking model. Fakes for offline runs and tests."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Protocol

import numpy as np
from openai import AsyncOpenAI


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

    async def embed(self, texts: list[str]) -> np.ndarray:
        response = await self.client.embeddings.create(model=self.embedding_model, input=texts)
        matrix = np.asarray([d.embedding for d in response.data], dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        return matrix

    async def structured(self, instruction: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.chat.completions.create(
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
            extra_body={
                "reasoning": {"effort": "low"},
                **({"provider": {"sort": self.provider_sort}} if self.provider_sort else {}),
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("ranking model returned an empty response")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            head = content[:160].replace("\n", "\\n")
            raise ValueError(
                f"ranking model returned malformed JSON ({len(content)} chars, "
                f"finish={response.choices[0].finish_reason}): {head!r}"
            ) from exc
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
            result = result[0]
        if not isinstance(result, dict):
            raise TypeError("ranking model returned JSON of an unsupported shape")
        return result
