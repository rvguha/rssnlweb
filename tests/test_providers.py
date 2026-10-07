"""OpenRouter ranking calls against a fake client."""

from types import SimpleNamespace

import pytest

from qdrss.providers import OpenRouter


def reply(content, *, finish="stop", provider="Groq"):
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)
    return SimpleNamespace(choices=[choice], provider=provider)


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


def ranker(replies):
    r = OpenRouter("k", "https://x.test", "m", "e")
    r.client = FakeClient(replies)
    return r


async def test_provider_returning_no_content_is_excluded_on_retry():
    # Seen with Groq's gpt-oss-20b: finish_reason "error", no content, every time at temperature 0.
    r = ranker([reply(None, finish="error", provider="Groq"), reply('{"results": []}', provider="Novita")])
    assert await r.structured("i", {"q": "x"}) == {"results": []}
    first, second = r.client.calls
    assert first["extra_body"]["provider"] == {"sort": "throughput"}
    assert second["extra_body"]["provider"] == {"sort": "throughput", "ignore": ["Groq"]}


async def test_no_retry_when_the_first_call_has_content():
    r = ranker([reply('{"results": [1]}')])
    assert await r.structured("i", {}) == {"results": [1]}
    assert len(r.client.calls) == 1


async def test_still_empty_after_the_retry_is_an_error():
    r = ranker([reply(None, finish="error"), reply(None, finish="error", provider="Novita")])
    with pytest.raises(RuntimeError, match="empty response"):
        await r.structured("i", {})
    assert len(r.client.calls) == 2


async def test_unknown_provider_cannot_be_excluded():
    r = ranker([reply(None, finish="error", provider=None)])
    with pytest.raises(RuntimeError, match="empty response"):
        await r.structured("i", {})
    assert len(r.client.calls) == 1
