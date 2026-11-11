"""Retry and rate-limit behaviour of the Gemini client (no network: the SDK calls are faked)."""

import numpy as np
import pytest

from placidbot import llm as llm_module
from placidbot.llm import (
    CHAT_MAX_WAIT_S, INGEST_MAX_WAIT_S, GeminiClient, LLMError, retry_hint_seconds, retry_wait_seconds,
)

QUOTA_MESSAGE = "Quota exceeded for embed_content_free_tier_requests, limit: 100. Please retry in 18.62s."


def test_retry_wait_uses_the_servers_hint_plus_a_margin():
    assert retry_wait_seconds(QUOTA_MESSAGE, fallback_s=2, max_wait_s=70) == pytest.approx(19.62)


def test_retry_wait_falls_back_to_backoff_without_a_hint():
    assert retry_wait_seconds("503 UNAVAILABLE", fallback_s=4, max_wait_s=70) == 4


def test_retry_wait_is_capped_so_a_visitor_never_waits_long():
    assert retry_wait_seconds("Please retry in 45s", fallback_s=2, max_wait_s=CHAT_MAX_WAIT_S) == CHAT_MAX_WAIT_S


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeResult:
    def __init__(self, count, dim):
        self.embeddings = [_FakeEmbedding([1.0] * dim) for _ in range(count)]


def _quota_error():
    from google.genai import errors

    return errors.APIError(429, {"error": {"code": 429, "message": QUOTA_MESSAGE, "status": "RESOURCE_EXHAUSTED"}})


def _client_with_fake_embed(monkeypatch, failures_before_success: int):
    """A GeminiClient whose embed_content hits the quota on the second batch."""
    client = GeminiClient("test-key", "answer-model", "embedding-model", embedding_dim=4)
    sleeps: list[float] = []
    calls: list[int] = []
    remaining = {"failures": failures_before_success}

    def fake_embed_content(*, model, contents, config):
        calls.append(len(contents))
        if len(calls) > 1 and remaining["failures"] > 0:
            remaining["failures"] -= 1
            raise _quota_error()
        return _FakeResult(len(contents), 4)

    monkeypatch.setattr(client._client.models, "embed_content", fake_embed_content)
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    return client, sleeps, calls


def test_document_embedding_waits_out_the_per_minute_quota(monkeypatch):
    client, sleeps, calls = _client_with_fake_embed(monkeypatch, failures_before_success=1)

    vectors = client.embed([f"chunk {i}" for i in range(150)], task="RETRIEVAL_DOCUMENT")

    assert vectors.shape == (150, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert calls == [100, 50, 50]  # second batch retried once
    assert sleeps == [pytest.approx(19.62)]
    assert sleeps[0] <= INGEST_MAX_WAIT_S


def test_query_embedding_fails_fast_instead_of_blocking_the_chat(monkeypatch):
    client, sleeps, _ = _client_with_fake_embed(monkeypatch, failures_before_success=99)
    client.embed(["warm-up call that succeeds"], task="RETRIEVAL_QUERY")

    with pytest.raises(LLMError):
        client.embed(["how much does it cost"], task="RETRIEVAL_QUERY")

    assert sleeps and all(wait <= CHAT_MAX_WAIT_S for wait in sleeps)


# -- model fallback when a free-tier quota is used up -----------------------------------------

DAILY_QUOTA_MESSAGE = "Generate content requests per day limit exceeded on Free Tier. Please retry in 4h13m57.8s."


def test_retry_hint_reads_seconds_and_hours_minutes_seconds():
    assert retry_hint_seconds(QUOTA_MESSAGE) == pytest.approx(18.62)
    assert retry_hint_seconds(DAILY_QUOTA_MESSAGE) == pytest.approx(4 * 3600 + 13 * 60 + 57.8)
    assert retry_hint_seconds("503 UNAVAILABLE") is None


class _FakeTextResponse:
    def __init__(self, text):
        self.text = text


def _api_error(code: int, message: str):
    from google.genai import errors

    return errors.APIError(code, {"error": {"code": code, "message": message, "status": "ERROR"}})


def _client_with_fake_generate(monkeypatch, behaviour):
    """`behaviour(model, call_number)` returns a JSON string or raises an API error."""
    client = GeminiClient("test-key", "model-a", "embedding-model", fallbacks={"model-a": ("model-b", "model-c")})
    calls: list[str] = []
    sleeps: list[float] = []

    def fake_generate_content(*, model, contents, config):
        calls.append(model)
        return _FakeTextResponse(behaviour(model, len(calls)))

    monkeypatch.setattr(client._client.models, "generate_content", fake_generate_content)
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    return client, calls, sleeps


def _ask(client: GeminiClient) -> dict:
    return client.generate_json(system="s", prompt="p", schema={"type": "object"})


def test_generate_moves_to_the_next_model_when_the_quota_is_used_up(monkeypatch):
    def behaviour(model, _):
        if model == "model-a":
            raise _api_error(429, DAILY_QUOTA_MESSAGE)
        return '{"answered_by": "%s"}' % model

    client, calls, sleeps = _client_with_fake_generate(monkeypatch, behaviour)

    assert _ask(client) == {"answered_by": "model-b"}
    assert calls == ["model-a", "model-b"]
    assert sleeps == []  # the visitor does not wait: the next model is tried at once
    assert client.last_model == "model-b"


def test_a_rate_limited_model_is_skipped_until_its_quota_resets(monkeypatch):
    def behaviour(model, _):
        if model == "model-a":
            raise _api_error(429, DAILY_QUOTA_MESSAGE)
        return "{}"

    client, calls, _ = _client_with_fake_generate(monkeypatch, behaviour)
    _ask(client)
    _ask(client)

    assert calls == ["model-a", "model-b", "model-b"]  # model-a is not asked again


def test_generate_fails_cleanly_when_every_model_is_rate_limited(monkeypatch):
    def behaviour(model, _):
        raise _api_error(429, DAILY_QUOTA_MESSAGE)

    client, calls, _ = _client_with_fake_generate(monkeypatch, behaviour)

    with pytest.raises(LLMError, match="rate limited"):
        _ask(client)
    assert calls == ["model-a", "model-b", "model-c"]
    with pytest.raises(LLMError, match="rate limited"):
        _ask(client)
    assert len(calls) == 3  # nothing is called while all models cool down


def test_server_errors_are_retried_on_the_same_model(monkeypatch):
    def behaviour(model, call_number):
        if call_number == 1:
            raise _api_error(503, "The model is overloaded.")
        return '{"ok": true}'

    client, calls, sleeps = _client_with_fake_generate(monkeypatch, behaviour)

    assert _ask(client) == {"ok": True}
    assert calls == ["model-a", "model-a"]
    assert sleeps == [2.0]


def test_thinking_level_is_sent_and_dropped_for_models_that_reject_it(monkeypatch):
    client, calls, _ = _client_with_fake_generate(monkeypatch, lambda model, _: "{}")
    client.thinking_level = "LOW"
    levels: list[str | None] = []

    def fake_generate_content(*, model, contents, config):
        level = config.thinking_config.thinking_level if config.thinking_config else None
        levels.append(level)
        if level is not None and model == "model-a":
            raise _api_error(400, "Thinking level LOW is not supported for this model.")
        return _FakeTextResponse("{}")

    monkeypatch.setattr(client._client.models, "generate_content", fake_generate_content)

    assert _ask(client) == {}
    assert _ask(client) == {}
    assert [str(level) if level else None for level in levels] == ["ThinkingLevel.LOW", None, None]


def test_from_settings_wires_models_and_fallbacks(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_ANSWER_MODEL", "big")
    monkeypatch.setenv("GEMINI_ANSWER_FALLBACKS", "medium, small")
    monkeypatch.setenv("GEMINI_ROUTER_MODEL", "tiny")
    monkeypatch.setenv("GEMINI_ROUTER_FALLBACKS", "")
    from placidbot.config import get_settings

    client = GeminiClient.from_settings(get_settings())

    assert client.answer_model == "big"
    assert client.candidates("big") == ["big", "medium", "small"]
    assert client.candidates("tiny")[0] == "tiny"
