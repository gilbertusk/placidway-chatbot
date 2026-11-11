"""All settings in one place, read from environment variables.

Secrets (the Gemini API key, the admin token) only ever come from the
environment. On Hugging Face Spaces they are set as Space secrets; locally
they come from a `.env` file that is git-ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: load a local .env when python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a dev convenience only
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_list(name: str, default: str) -> tuple[str, ...]:
    """Comma-separated list. Unset = `default`; set to an empty string = no entries."""
    raw = os.getenv(name)
    return tuple(item.strip() for item in (default if raw is None else raw).split(",") if item.strip())


# Tried in order when the preferred model's free-tier quota is used up (quotas are per model).
DEFAULT_ANSWER_FALLBACKS = "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"
DEFAULT_ROUTER_FALLBACKS = "gemini-3.1-flash-lite,gemini-3.5-flash,gemini-3.6-flash"


@dataclass(frozen=True)
class Settings:
    # LLM provider
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip(), repr=False)
    answer_model: str = field(default_factory=lambda: _env("GEMINI_ANSWER_MODEL", "gemini-3.8-flash"))
    answer_fallbacks: tuple[str, ...] = field(
        default_factory=lambda: _env_list("GEMINI_ANSWER_FALLBACKS", DEFAULT_ANSWER_FALLBACKS))
    router_model: str = field(default_factory=lambda: _env("GEMINI_ROUTER_MODEL", "gemini-3.5-flash-lite"))
    router_fallbacks: tuple[str, ...] = field(
        default_factory=lambda: _env_list("GEMINI_ROUTER_FALLBACKS", DEFAULT_ROUTER_FALLBACKS))
    embedding_model: str = field(default_factory=lambda: _env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"))
    embedding_dim: int = field(default_factory=lambda: int(_env("EMBEDDING_DIM", "768")))

    # Crawler etiquette
    crawler_contact: str = field(default_factory=lambda: _env("CRAWLER_CONTACT", "gilbertuskristo@gmail.com"))
    request_interval_s: float = field(default_factory=lambda: float(_env("REQUEST_INTERVAL_S", "1.0")))
    request_timeout_s: float = field(default_factory=lambda: float(_env("REQUEST_TIMEOUT_S", "30")))

    # Retrieval
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "6")))
    chunk_max_chars: int = field(default_factory=lambda: int(_env("CHUNK_MAX_CHARS", "1400")))

    # Admin (refresh + unanswered log). Empty token = admin actions disabled in the UI.
    admin_token: str = field(default_factory=lambda: os.getenv("ADMIN_TOKEN", "").strip(), repr=False)

    # Storage
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT_DIR / "data"))))

    @property
    def user_agent(self) -> str:
        return (
            "PlacidWayKnowledgeBot/1.0 (AI Automation Assessment crawler; "
            f"fetches 7 assigned pages only; contact: {self.crawler_contact})"
        )

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"


def get_settings() -> Settings:
    return Settings()
