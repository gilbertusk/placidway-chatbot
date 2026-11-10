"""The only module that talks to the LLM provider (Google Gemini).

Everything else depends on the small `LLMClient` interface below, so the
provider can be swapped (or faked in tests) without touching the bot logic.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The model could not be reached or returned something unusable."""


class LLMClient(Protocol):
    def generate_json(self, *, system: str, prompt: str, schema: dict, model: str | None = None) -> dict: ...

    def embed(self, texts: list[str], *, task: str) -> np.ndarray: ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class GeminiClient:
    """Gemini via the official `google-genai` SDK."""

    EMBED_BATCH = 100  # API limit per embed request

    def __init__(self, api_key: str, answer_model: str, embedding_model: str, embedding_dim: int = 768, retries: int = 3):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        from google import genai  # imported here so tests do not need the SDK configured

        self._client = genai.Client(api_key=api_key)
        self.answer_model = answer_model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.retries = retries

    def _with_retries(self, fn):
        from google.genai import errors

        delay = 2.0
        for attempt in range(self.retries + 1):
            try:
                return fn()
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                retryable = code in (429, 500, 502, 503, 504)
                if not retryable or attempt == self.retries:
                    raise LLMError(f"Gemini API error {code}: {exc}") from exc
                log.warning("Gemini error %s, retrying in %.0fs", code, delay)
                time.sleep(delay)
                delay *= 2
        raise LLMError("unreachable")

    def generate_json(self, *, system: str, prompt: str, schema: dict, model: str | None = None) -> dict:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            response_mime_type="application/json",
            response_json_schema=schema,
        )
        response = self._with_retries(
            lambda: self._client.models.generate_content(model=model or self.answer_model, contents=prompt, config=config)
        )
        text = (response.text or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model did not return valid JSON: {text[:200]!r}") from exc

    def embed(self, texts: list[str], *, task: str) -> np.ndarray:
        """task: "RETRIEVAL_DOCUMENT" for chunks, "RETRIEVAL_QUERY" for questions."""
        from google.genai import types

        config = types.EmbedContentConfig(task_type=task, output_dimensionality=self.embedding_dim)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.EMBED_BATCH):
            batch = texts[start:start + self.EMBED_BATCH]
            result = self._with_retries(
                lambda: self._client.models.embed_content(model=self.embedding_model, contents=batch, config=config)
            )
            vectors.extend(e.values for e in result.embeddings)
        return _normalise(np.asarray(vectors, dtype=np.float32))
