import time

import pytest
from conftest import FakeLLM, fixture_html

from placidbot.crawler import FetchError, FetchResult, PoliteCrawler
from placidbot.index import HybridIndex
from placidbot.ingest import build_index
from placidbot.sources import SOURCES_BY_ID

FIXTURE_FOR = {
    "adenocarcinoma-package": "package.html",
    "itc-price-list": "price_list.html",
    "alt-medicine-prices": "comparison.html",
}
TEST_SOURCES = tuple(SOURCES_BY_ID[s] for s in FIXTURE_FOR)


class FixtureCrawler:
    """Serves fixture HTML instead of hitting the network."""

    def __init__(self, fail: set[str] | None = None, html_override: dict | None = None):
        self.fail = fail or set()
        self.html_override = html_override or {}
        self.fetched = []

    def fetch(self, url):
        source = next(s for s in TEST_SOURCES if s.url == url)
        self.fetched.append(source.id)
        if source.id in self.fail:
            raise FetchError("HTTP 503")
        html = self.html_override.get(source.id) or fixture_html(FIXTURE_FOR[source.id])
        return FetchResult(url=url, status=200, html=html)


def test_build_index_writes_snapshots_and_index(settings):
    index, report = build_index(settings, FakeLLM(), FixtureCrawler(), TEST_SOURCES)
    assert report.pages_ok == list(FIXTURE_FOR)
    assert report.embeddings and index.embeddings is not None
    assert (settings.pages_dir / "adenocarcinoma-package.md").read_text().count("$30,000") == 1
    loaded = HybridIndex.load(settings.index_dir)
    assert len(loaded) == len(index) and loaded.embeddings.shape[0] == len(index)


def test_refresh_reports_changes_and_keeps_failed_pages(settings):
    build_index(settings, FakeLLM(), FixtureCrawler(), TEST_SOURCES)

    _, unchanged = build_index(settings, FakeLLM(), FixtureCrawler(), TEST_SOURCES)
    assert unchanged.pages_changed == []

    new_html = fixture_html("price_list.html").replace("$18,995", "$19,500")
    index, changed = build_index(settings, FakeLLM(), FixtureCrawler(fail={"alt-medicine-prices"},
                                 html_override={"itc-price-list": new_html}), TEST_SOURCES)
    assert changed.pages_changed == ["itc-price-list"]
    assert "alt-medicine-prices" in changed.pages_failed
    # the failed page's previous content is still searchable
    assert any(c.source_id == "alt-medicine-prices" for c in index.chunks)
    assert any("$19,500" in c.text for c in index.chunks)


def test_keyword_only_index_without_llm(settings):
    index, report = build_index(settings, None, FixtureCrawler(), TEST_SOURCES)
    assert not report.embeddings and index.embeddings is None
    hits = index.search("ozone therapy price Turkey")
    assert hits and "Fizikon" in hits[0].chunk.text


def test_hybrid_search_finds_price_table(settings):
    llm = FakeLLM()
    index, _ = build_index(settings, llm, FixtureCrawler(), TEST_SOURCES)
    hits = index.search("How much does the adenocarcinoma in-patient program cost?", llm, k=3)
    assert "$30,000" in hits[0].chunk.text


class FakeResponse:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text
        self.encoding, self.apparent_encoding = "utf-8", "utf-8"


def _crawler_with(responses, interval=0.0):
    crawler = PoliteCrawler("TestBot/1.0", interval_s=interval, retries=0)
    calls = []

    def fake_get(url, timeout):
        calls.append((url, time.monotonic()))
        return responses(url)

    crawler._session.get = fake_get
    return crawler, calls


def test_crawler_respects_robots_txt():
    def responses(url):
        if url.endswith("/robots.txt"):
            return FakeResponse(200, "User-agent: *\nDisallow: /private/")
        return FakeResponse(200, "<html>ok</html>")

    crawler, _ = _crawler_with(responses)
    assert crawler.fetch("https://site.test/public/page").html == "<html>ok</html>"
    with pytest.raises(FetchError, match="robots.txt"):
        crawler.fetch("https://site.test/private/page")


def test_crawler_waits_between_requests():
    crawler, calls = _crawler_with(lambda url: FakeResponse(404 if url.endswith("robots.txt") else 200, "<html/>"), interval=0.2)
    crawler.fetch("https://site.test/a")
    crawler.fetch("https://site.test/b")
    gaps = [b[1] - a[1] for a, b in zip(calls, calls[1:])]
    assert len(calls) == 3 and min(gaps) >= 0.15  # 0.2 s interval; leave room for coarse OS timers


def test_crawler_sends_user_agent():
    crawler = PoliteCrawler("PlacidWayKnowledgeBot/1.0 (contact: x@y.z)")
    assert crawler._session.headers["User-Agent"].startswith("PlacidWayKnowledgeBot/1.0")
