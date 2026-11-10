"""Shared test helpers: a fake LLM (no network, no API key) and fixture pages."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures"


def fixture_html(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class FakeLLM:
    """Deterministic stand-in for Gemini.

    * embed(): hashed bag-of-words vectors, so similar wording -> similar vectors.
    * generate_json(): the router returns `route`; the answerer pops scripted
      answers from `answers` (or calls `answer_fn(prompt)`).
    """

    def __init__(self, route: dict | None = None, answers: list[dict] | None = None, answer_fn=None):
        self.route = route or {"standalone_question": "", "language": "en", "intent": "question"}
        self.answers = list(answers or [])
        self.answer_fn = answer_fn
        self.answer_prompts: list[str] = []
        self.embed_calls = 0

    def generate_json(self, *, system: str, prompt: str, schema: dict, model: str | None = None) -> dict:
        if "standalone_question" in schema["properties"]:
            route = dict(self.route)
            if not route.get("standalone_question"):
                route["standalone_question"] = prompt.rsplit("LATEST USER MESSAGE:", 1)[-1].strip()
            return route
        self.answer_prompts.append(prompt)
        if self.answer_fn:
            return self.answer_fn(prompt)
        return self.answers.pop(0)

    def embed(self, texts: list[str], *, task: str) -> np.ndarray:
        self.embed_calls += 1
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                vectors[i, hash(token) % 64] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ADMIN_TOKEN", "test-token")
    monkeypatch.setenv("REQUEST_INTERVAL_S", "0")
    from placidbot.config import get_settings

    return get_settings()
