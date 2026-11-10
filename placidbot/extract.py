"""Turn a fetched HTML page into clean, structured Markdown sections.

Pipeline for one page:
1. Read the page title and any schema.org price data (JSON-LD offers).
2. Strip everything that is not content: scripts, navigation, headers,
   footers, forms, modals, share/social widgets.
3. Find the main content block (the <main>/<article> element, or the
   smallest ancestor of the <h1> that holds most of the page text).
4. Convert it to Markdown, which keeps headings, lists and price tables.
5. Split the Markdown into sections by heading, so each section knows its
   heading path (e.g. "Cost of ... > Included in the Out-Patient Program").

Boilerplate repeated across many pages (e.g. a "Related Experiences" block)
is removed later, in `drop_cross_page_boilerplate`, because it can only be
recognised by comparing pages.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

REMOVE_TAGS = [
    "script", "style", "noscript", "svg", "iframe", "form", "button", "input",
    "select", "textarea", "nav", "footer", "header", "template", "canvas", "dialog",
]

# class/id tokens that mark page chrome rather than content
BOILERPLATE_TOKEN = re.compile(
    r"^(nav|navbar|menu|main-menu|mega-menu|footer|site-footer|site-header|top-bar|topbar|"
    r"breadcrumb|breadcrumbs|cookie|cookies|cookie-banner|modal|popup|newsletter|"
    r"social|social-share|share|sharethis|subscribe|back-to-top|skip-link)$",
    re.I,
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Section:
    heading_path: list[str]
    text: str


@dataclass
class ExtractedPage:
    url: str
    title: str
    sections: list[Section] = field(default_factory=list)
    structured_prices: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Readable snapshot of what the bot actually knows about this page."""
        lines = [f"# {self.title}", "", f"Source: {self.url}", ""]
        if self.structured_prices:
            lines += ["## Structured price data (schema.org)", ""]
            lines += [f"- {p}" for p in self.structured_prices]
            lines.append("")
        for sec in self.sections:
            lines.append("## " + " > ".join(sec.heading_path))
            lines.append("")
            lines.append(sec.text)
            lines.append("")
        return "\n".join(lines).strip() + "\n"


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _page_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1 and _clean_text(h1.get_text()):
        return _clean_text(h1.get_text())
    if soup.title and soup.title.string:
        return _clean_text(soup.title.string)
    return "Untitled page"


def _json_ld_prices(soup: BeautifulSoup) -> list[str]:
    """Collect price facts published as schema.org Offer data, if any."""
    facts: list[str] = []

    def walk(node, context_name: str = "") -> None:
        if isinstance(node, list):
            for item in node:
                walk(item, context_name)
            return
        if not isinstance(node, dict):
            return
        name = node.get("name") or context_name
        types = node.get("@type")
        types = types if isinstance(types, list) else [types]
        if any(t in ("Offer", "AggregateOffer") for t in types):
            currency = node.get("priceCurrency", "")
            if node.get("price") not in (None, ""):
                facts.append(f"{name}: {node['price']} {currency}".strip())
            elif node.get("lowPrice") not in (None, ""):
                high = node.get("highPrice")
                rng = f"{node['lowPrice']} to {high}" if high else f"from {node['lowPrice']}"
                facts.append(f"{name}: {rng} {currency}".strip())
        for key, value in node.items():
            if key in ("offers", "@graph", "itemListElement", "item", "makesOffer", "hasOfferCatalog"):
                walk(value, name if isinstance(name, str) else "")

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            walk(json.loads(tag.string or ""))
        except (json.JSONDecodeError, TypeError):
            continue
    # keep order, drop duplicates
    return list(dict.fromkeys(f for f in facts if f))


def _is_boilerplate(tag: Tag) -> bool:
    if tag.attrs is None:  # already decomposed
        return False
    tokens = list(tag.get("class") or [])
    if tag.get("id"):
        tokens.append(tag["id"])
    if tag.get("role") in ("navigation", "banner", "contentinfo", "dialog"):
        return True
    if tag.get("aria-hidden") == "true":
        return True
    return any(BOILERPLATE_TOKEN.match(t) for t in tokens)


def _strip_chrome(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()
    for tag in soup.find_all(True):
        if _is_boilerplate(tag):
            tag.decompose()
    for tag in soup.find_all("img"):
        tag.decompose()


def _main_block(soup: BeautifulSoup) -> Tag:
    for selector in ("main", "article", "[role=main]"):
        found = soup.select_one(selector)
        if found and len(found.get_text(strip=True)) > 200:
            return found
    body = soup.body or soup
    total = len(body.get_text(" ", strip=True)) or 1
    h1 = soup.find("h1")
    if h1 is None:
        return body
    node: Tag = h1
    while node.parent is not None and node.parent is not body:
        node = node.parent
        if len(node.get_text(" ", strip=True)) >= 0.5 * total:
            return node
    return body


def _to_markdown(block: Tag) -> str:
    md = markdownify(str(block), heading_style="ATX", bullets="-", strip=["a", "span"])
    lines = []
    for raw in md.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("|"):
            # markdownify pads table cells; collapse the inner whitespace
            line = re.sub(r"[ \t]+", " ", stripped)
        else:
            line = re.sub(r"[ \t]+", " ", stripped)
        lines.append(line)
    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _split_sections(markdown: str, title: str) -> list[Section]:
    sections: list[Section] = []
    path: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            heading_path = [h for _, h in path] or [title]
            sections.append(Section(heading_path=heading_path, text=text))
        buffer.clear()

    for line in markdown.splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading = _clean_text(match.group(2).strip("#*_ "))
            if not heading:
                continue
            path = [(lvl, h) for lvl, h in path if lvl < level]
            path.append((level, heading))
        else:
            buffer.append(line)
    flush()
    return sections


def extract_page(html: str, url: str) -> ExtractedPage:
    soup = BeautifulSoup(html, "lxml")
    title = _page_title(soup)
    prices = _json_ld_prices(soup)  # read before scripts are stripped
    _strip_chrome(soup)
    markdown = _to_markdown(_main_block(soup))

    if len(markdown) < 300:
        # Very little survived: fall back to a general-purpose extractor.
        import trafilatura

        fallback = trafilatura.extract(html, include_tables=True, output_format="markdown", url=url)
        if fallback and len(fallback) > len(markdown):
            markdown = fallback

    return ExtractedPage(url=url, title=title, sections=_split_sections(markdown, title), structured_prices=prices)


_NUMERIC = re.compile(r"\d")


def drop_cross_page_boilerplate(pages: list[ExtractedPage], min_pages: int = 4) -> None:
    """Remove lines that appear on `min_pages` or more pages.

    Such lines are template text (calls to action, "Related Experiences",
    review widgets). Lines with digits and table rows are kept, because a
    price repeated on several package pages is real content, not chrome.
    Sections left empty are dropped. Mutates the pages in place.
    """
    seen_on: Counter[str] = Counter()
    for page in pages:
        page_lines = {
            line.strip()
            for sec in page.sections
            for line in sec.text.splitlines()
            if line.strip()
        }
        seen_on.update(page_lines)

    repeated = {
        line for line, count in seen_on.items()
        if count >= min_pages and not _NUMERIC.search(line) and not line.startswith("|")
    }
    if not repeated:
        return
    for page in pages:
        kept_sections = []
        for sec in page.sections:
            lines = [ln for ln in sec.text.splitlines() if ln.strip() not in repeated]
            text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
            if text:
                kept_sections.append(Section(heading_path=sec.heading_path, text=text))
        page.sections = kept_sections
