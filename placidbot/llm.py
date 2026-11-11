"""The only module that talks to the LLM provider (Google Gemini).

Everything else depends on the small `LLMClient` interface below, so the
provider can be swapped (or faked in tests) without touching the bot logic.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, Sequence, TypeVar

import numpy as np

if TYPE_CHECKING:
    from .config import Settings

log = logging.getLogger(__name__)

T = TypeVar("T")

RATE_LIMIT_STATUS = 429
RETRYABLE_STATUS = (RATE_LIMIT_STATUS, 500, 502, 503, 504)
# "Please retry in 18.6s" (per-minute quota) or "Please retry in 4h13m57.8s" (per-day quota)
RETRY_HINT_RE = re.compile(r"retry in\s+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", re.IGNORECASE)
RETRY_HINT_MARGIN_S = 1.0
CHAT_MAX_WAIT_S = 8.0  # a visitor is waiting: give up quickly and show a friendly message
INGEST_MAX_WAIT_S = 70.0  # building the index can sit out a per-minute quota window
DEFAULT_COOLDOWN_S = 60.0  # how long to leave a rate-limited model alone when the error gives no hint
ERROR_TEXT_LIMIT = 300
# "Thinking level MINIMAL is not supported for this model."
THINKING_UNSUPPORTED_RE = re.compile(r"error 400\b.*thinking", re.IGNORECASE | re.DOTALL)


def retry_hint_seconds(error_text: str) -> float | None:
    """The wait the server asks for in a rate-limit error, or None if it gives none."""
    match = RETRY_HINT_RE.search(error_text or "")
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0)


def retry_wait_seconds(error_text: str, fallback_s: float, max_wait_s: float) -> float:
    """Seconds to wait before retrying a failed call.

    Rate-limit errors say when the quota frees up ("Please retry in 18.6s").
    Waiting that long (plus a margin) succeeds; a short fixed back-off does
    not. Without a hint, use `fallback_s`. Never wait longer than `max_wait_s`.
    """
    hint = retry_hint_seconds(error_text)
    wait = hint + RETRY_HINT_MARGIN_S if hint is not None else fallback_s
    return min(wait, max_wait_s)


class LLMError(RuntimeError):
    """The model could not be reached or returned something unusable."""


class RateLimited(LLMError):
    """This model's quota is used up for now; another model may still answer."""


class LLMClient(Protocol):
    def generate_json(self, *, system: str, prompt: str, schema: dict, model: str | None = None) -> dict: ...

    def embed(self, texts: list[str], *, task: str) -> np.ndarray: ...


def _parse_json(raw: str | None) -> dict:
    text = (raw or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model did not return valid JSON: {text[:200]!r}") from exc


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class GeminiClient:
    """Gemini via the official `google-genai` SDK.

    Free-tier quotas are counted per model and are small (20 answers a day on
    the newest flash model when this was written). `fallbacks` maps a model to
    the models to try next when its quota is used up, so the bot keeps
    answering instead of telling visitors to come back tomorrow.
    """

    EMBED_BATCH = 100  # API limit per embed request

    def __init__(self, api_key: str, answer_model: str, embedding_model: str, embedding_dim: int = 768,
                 retries: int = 3, fallbacks: Mapping[str, Sequence[str]] | None = None, thinking_level: str = ""):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        from google import genai  # imported here so tests do not need the SDK configured

        self._client = genai.Client(api_key=api_key)
        self.answer_model = answer_model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.retries = retries
        self.thinking_level = thinking_level
        self.fallbacks = {model: tuple(others) for model, others in (fallbacks or {}).items()}
        self.last_model: str | None = None  # which model wrote the latest reply (shown in test reports)
        self._cooldown_until: dict[str, float] = {}  # model -> time.monotonic() when its quota frees up
        self._no_thinking_level: set[str] = set()  # models that rejected `thinking_level`

    @classmethod
    def from_settings(cls, settings: Settings) -> "GeminiClient":
        return cls(
            settings.gemini_api_key, settings.answer_model, settings.embedding_model, settings.embedding_dim,
            fallbacks={
                settings.answer_model: settings.answer_fallbacks,
                settings.router_model: settings.router_fallbacks,
            },
            thinking_level=settings.thinking_level,
        )

    def candidates(self, model: str) -> list[str]:
        """`model` followed by its fallbacks, without duplicates."""
        return list(dict.fromkeys((model, *self.fallbacks.get(model, ()))))

    def _with_retries(self, fn: Callable[[], T], max_wait_s: float = CHAT_MAX_WAIT_S,
                      cooldown_model: str | None = None) -> T:
        """Run one API call, retrying transient failures with back-off.

        With `cooldown_model`, a rate limit is not waited out: the model is
        set aside until its quota resets and RateLimited is raised at once,
        so the caller can try another model.
        """
        from google.genai import errors

        backoff = 2.0
        for attempt in range(self.retries + 1):
            try:
                return fn()
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                detail = str(exc)
                if code == RATE_LIMIT_STATUS and cooldown_model is not None:
                    hint = retry_hint_seconds(detail)
                    cooldown = hint + RETRY_HINT_MARGIN_S if hint is not None else DEFAULT_COOLDOWN_S
                    self._cooldown_until[cooldown_model] = time.monotonic() + cooldown
                    raise RateLimited(f"{cooldown_model} is rate limited for {cooldown:.0f}s") from exc
                if code not in RETRYABLE_STATUS or attempt == self.retries:
                    raise LLMError(f"Gemini API error {code}: {detail[:ERROR_TEXT_LIMIT]}") from exc
                wait = retry_wait_seconds(detail, backoff, max_wait_s)
                log.warning("Gemini error %s, retrying in %.0fs", code, wait)
                time.sleep(wait)
                backoff *= 2
        raise LLMError("unreachable")

    def generate_json(self, *, system: str, prompt: str, schema: dict, model: str | None = None) -> dict:
        from google.genai import types

        def config_for(candidate: str) -> types.GenerateContentConfig:
            level = "" if candidate in self._no_thinking_level else self.thinking_level
            return types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.0,
                response_mime_type="application/json",
                response_json_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_level=level) if level else None,
            )

        now = time.monotonic()
        ready = [m for m in self.candidates(model or self.answer_model) if self._cooldown_until.get(m, 0.0) <= now]
        for candidate in ready:
            call = lambda: self._client.models.generate_content(  # noqa: E731
                model=candidate, contents=prompt, config=config_for(candidate))
            try:
                try:
                    response = self._with_retries(call, cooldown_model=candidate)
                except LLMError as exc:
                    if candidate in self._no_thinking_level or not THINKING_UNSUPPORTED_RE.search(str(exc)):
                        raise
                    log.warning("%s does not accept thinking level %s; using its default", candidate, self.thinking_level)
                    self._no_thinking_level.add(candidate)
                    response = self._with_retries(call, cooldown_model=candidate)
            except RateLimited as exc:
                log.warning("%s; trying the next model", exc)
                continue
            self.last_model = candidate
            return _parse_json(response.text)
        raise LLMError("All configured Gemini models are rate limited right now")

    def embed(self, texts: list[str], *, task: str) -> np.ndarray:
        """task: "RETRIEVAL_DOCUMENT" for chunks, "RETRIEVAL_QUERY" for questions.

        The free tier allows 100 embedded texts per minute, so indexing more
        than 100 chunks hits the limit. Indexing waits for the quota to free
        up; a visitor's question never waits that long.
        """
        from google.genai import types

        config = types.EmbedContentConfig(task_type=task, output_dimensionality=self.embedding_dim)
        max_wait_s = INGEST_MAX_WAIT_S if task == "RETRIEVAL_DOCUMENT" else CHAT_MAX_WAIT_S
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.EMBED_BATCH):
            batch = texts[start:start + self.EMBED_BATCH]
            result = self._with_retries(
                lambda: self._client.models.embed_content(model=self.embedding_model, contents=batch, config=config),
                max_wait_s,
            )
            vectors.extend(e.values for e in result.embeddings)
        return _normalise(np.asarray(vectors, dtype=np.float32))
