"""Content ingestion and refresh: fetch -> extract -> chunk -> embed -> save.

Run from the command line to (re)build the knowledge base:

    python -m placidbot.ingest

The chat UI's admin "Refresh content" button calls the same `build_index`.
Each run writes:
* data/pages/<source>.md  readable snapshot of what was extracted per page
* data/index/              chunks, embeddings and metadata used by the bot
and reports which pages changed since the previous run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .chunker import Chunk, chunk_page
from .config import Settings, get_settings
from .crawler import FetchError, LocalHtmlCrawler, PoliteCrawler
from .extract import ExtractedPage, drop_cross_page_boilerplate, extract_page
from .index import HybridIndex
from .llm import LLMClient, LLMError
from .sources import SOURCES, Source

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    built_at: str
    pages_ok: list[str] = field(default_factory=list)
    pages_failed: dict[str, str] = field(default_factory=dict)
    pages_changed: list[str] = field(default_factory=list)
    chunk_count: int = 0
    embeddings: bool = False
    pages_total: int = 0

    def summary(self) -> str:
        lines = [
            f"Built at {self.built_at} UTC: {len(self.pages_ok)}/{self.pages_total} pages fetched, "
            f"{self.chunk_count} chunks, embeddings {'on' if self.embeddings else 'off (keyword search only)'}."
        ]
        if self.pages_changed:
            lines.append("Changed since last refresh: " + ", ".join(self.pages_changed))
        else:
            lines.append("No page content changed since the last refresh.")
        for source_id, error in self.pages_failed.items():
            lines.append(f"FAILED {source_id}: {error} (previous content for this page was kept)")
        return "\n".join(lines)


def _content_hash(page: ExtractedPage) -> str:
    return hashlib.sha256(page.to_markdown().encode()).hexdigest()


def build_index(settings: Settings | None = None, llm: LLMClient | None = None,
                crawler: PoliteCrawler | None = None, sources: tuple[Source, ...] = SOURCES) -> tuple[HybridIndex, IngestReport]:
    settings = settings or get_settings()
    crawler = crawler or PoliteCrawler(settings.user_agent, settings.request_interval_s, settings.request_timeout_s)
    report = IngestReport(built_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), pages_total=len(sources))

    previous = HybridIndex.load(settings.index_dir)
    previous_hashes: dict[str, str] = (previous.meta.get("page_hashes", {}) if previous else {})

    # 1. fetch + extract
    pages: dict[str, ExtractedPage] = {}
    for source in sources:
        try:
            result = crawler.fetch(source.url)
            pages[source.id] = extract_page(result.html, source.url)
            report.pages_ok.append(source.id)
            log.info("Fetched %s", source.id)
        except FetchError as exc:
            report.pages_failed[source.id] = str(exc)
            log.error("Failed %s: %s", source.id, exc)

    drop_cross_page_boilerplate(list(pages.values()))

    # 2. chunk (reuse previous chunks for pages that failed to fetch this time)
    chunks: list[Chunk] = []
    page_hashes: dict[str, str] = {}
    settings.pages_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        if source.id in pages:
            page = pages[source.id]
            (settings.pages_dir / f"{source.id}.md").write_text(page.to_markdown(), encoding="utf-8")
            page_hashes[source.id] = _content_hash(page)
            if previous_hashes.get(source.id) != page_hashes[source.id]:
                report.pages_changed.append(source.id)
            chunks.extend(chunk_page(page, source, settings.chunk_max_chars))
        elif previous is not None:
            chunks.extend(c for c in previous.chunks if c.source_id == source.id)
            if source.id in previous_hashes:
                page_hashes[source.id] = previous_hashes[source.id]

    if not chunks:
        raise RuntimeError("No content could be ingested: " + json.dumps(report.pages_failed))

    # 3. embed (optional: the bot still works on keyword search without it)
    embeddings = None
    if llm is not None:
        try:
            embeddings = llm.embed([c.text for c in chunks], task="RETRIEVAL_DOCUMENT")
            report.embeddings = True
        except LLMError as exc:
            log.error("Embedding failed, saving keyword-only index: %s", exc)

    report.chunk_count = len(chunks)
    meta = {
        "built_at": report.built_at,
        "page_hashes": page_hashes,
        "embedding_model": settings.embedding_model if embeddings is not None else None,
        "sources": [{"id": s.id, "type": s.page_type, "url": s.url} for s in sources],
    }
    index = HybridIndex(chunks, embeddings, meta)
    index.save(settings.index_dir)
    return index, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the PlacidWay pages and rebuild the knowledge base.")
    parser.add_argument("--no-embeddings", action="store_true", help="build a keyword-only index (no API key needed)")
    parser.add_argument("--html-dir", help="read saved pages <source-id>.html from this folder instead of fetching "
                                           "(fallback if the site blocks automated requests)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = get_settings()
    llm = None
    if not args.no_embeddings:
        from .llm import GeminiClient

        llm = GeminiClient(settings.gemini_api_key, settings.answer_model, settings.embedding_model, settings.embedding_dim)
    crawler = LocalHtmlCrawler(args.html_dir) if args.html_dir else None
    _, report = build_index(settings, llm, crawler)
    print(report.summary())


if __name__ == "__main__":
    main()
