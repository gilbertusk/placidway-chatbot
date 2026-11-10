"""A polite fetcher for the assigned pages.

Etiquette required by the brief:
* identify ourselves with a clear User-Agent,
* respect robots.txt,
* keep to roughly one request per second.

It fetches only the URLs it is given and never follows links.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests

log = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """A page could not be fetched (blocked by robots.txt, HTTP error, network)."""


@dataclass
class FetchResult:
    url: str
    status: int
    html: str


class PoliteCrawler:
    def __init__(self, user_agent: str, interval_s: float = 1.0, timeout_s: float = 30.0, retries: int = 2):
        self.user_agent = user_agent
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self.retries = retries
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent, "Accept-Language": "en"})
        self._last_request_at = 0.0
        self._robots: dict[str, RobotFileParser] = {}

    # -- rate limiting -------------------------------------------------
    def _wait_turn(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.interval_s:
            time.sleep(self.interval_s - elapsed)
        self._last_request_at = time.monotonic()

    def _get(self, url: str) -> requests.Response:
        self._wait_turn()
        return self._session.get(url, timeout=self.timeout_s)

    # -- robots.txt ----------------------------------------------------
    def _robots_for(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin in self._robots:
            return self._robots[origin]

        parser = RobotFileParser()
        robots_url = f"{origin}/robots.txt"
        try:
            resp = self._get(robots_url)
            if resp.status_code >= 500:
                # Server trouble: be conservative and treat everything as disallowed.
                parser.parse(["User-agent: *", "Disallow: /"])
            elif resp.status_code >= 400:
                # No robots.txt: everything is allowed (standard behaviour).
                parser.parse([])
            else:
                parser.parse(resp.text.splitlines())
        except requests.RequestException as exc:
            log.warning("Could not read %s (%s); treating site as disallowed", robots_url, exc)
            parser.parse(["User-agent: *", "Disallow: /"])
        self._robots[origin] = parser
        return parser

    def allowed(self, url: str) -> bool:
        return self._robots_for(url).can_fetch(self.user_agent, url)

    # -- fetching ------------------------------------------------------
    def fetch(self, url: str) -> FetchResult:
        if not self.allowed(url):
            raise FetchError(f"robots.txt disallows {url}")

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self._get(url)
                if resp.status_code == 200:
                    resp.encoding = resp.encoding or resp.apparent_encoding
                    return FetchResult(url=url, status=200, html=resp.text)
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = FetchError(f"HTTP {resp.status_code} for {url}")
                else:
                    raise FetchError(f"HTTP {resp.status_code} for {url}")
            except requests.RequestException as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(2 ** attempt)  # back off before retrying
        raise FetchError(f"Giving up on {url}: {last_error}")


class LocalHtmlCrawler:
    """Reads pages saved from a browser ("Save page as… HTML only") instead of
    fetching them. Files are named after the source id, e.g. itc-price-list.html.
    A fallback for when the site blocks automated requests."""

    def __init__(self, folder: str):
        from pathlib import Path

        from .sources import SOURCES

        self.folder = Path(folder)
        self._id_for_url = {s.url: s.id for s in SOURCES}

    def fetch(self, url: str) -> FetchResult:
        path = self.folder / f"{self._id_for_url.get(url, '')}.html"
        if not path.exists():
            raise FetchError(f"No saved file {path.name} for {url}")
        return FetchResult(url=url, status=200, html=path.read_text(encoding="utf-8", errors="replace"))
