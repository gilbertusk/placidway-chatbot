from conftest import fixture_html

from placidbot.chunker import chunk_page
from placidbot.extract import ExtractedPage, Section, drop_cross_page_boilerplate, extract_page
from placidbot.sources import SOURCES_BY_ID


def test_extract_keeps_content_and_drops_chrome():
    page = extract_page(fixture_html("package.html"), "https://example.com/package")
    md = page.to_markdown()

    assert page.title.startswith("Alternative Adenocarcinoma Cancer Treatment Package")
    assert "| Out-Patient Program | 3 Weeks | $18,995 |" in md
    assert "| In-Patient Program | 3 Weeks | $30,000 |" in md
    assert "Price starting from" in md
    assert "- Weekly lab work" in md
    # page chrome is gone
    for chrome in ("Destinations", "All rights reserved", "Home / Mexico", "dataLayer", "Request your free quote today", "Get a Free Quote"):
        assert chrome not in md, chrome


def test_extract_reads_schema_org_price():
    page = extract_page(fixture_html("package.html"), "https://example.com/package")
    assert page.structured_prices == ["Alternative Adenocarcinoma Cancer Treatment Package: 18995 USD"]


def test_sections_carry_heading_path():
    page = extract_page(fixture_html("package.html"), "https://example.com/package")
    paths = [" > ".join(s.heading_path) for s in page.sections]
    assert any(p.endswith("Cost of Adenocarcinoma Cancer Treatment in Tijuana, Mexico > Included in the Out-Patient Program ($18,995 USD)") for p in paths)


def test_cross_page_boilerplate_removed_but_prices_kept():
    def page(i):
        return ExtractedPage(url=f"u{i}", title=f"t{i}", sections=[Section(["t"], (
            f"Unique text for page {i} about something.\n"
            "Explore more medical travel experiences.\n"
            "| Out-Patient Program | 3 Weeks | $18,995 |\n"
            "Price starting from $18,995"
        ))])

    pages = [page(i) for i in range(5)]
    drop_cross_page_boilerplate(pages, min_pages=4)
    text = pages[0].sections[0].text
    assert "Explore more medical travel experiences." not in text
    assert "| Out-Patient Program | 3 Weeks | $18,995 |" in text
    assert "Price starting from $18,995" in text
    assert "Unique text for page 0" in text


def test_chunks_have_header_and_stay_within_sections():
    page = extract_page(fixture_html("package.html"), "https://example.com/package")
    chunks = chunk_page(page, SOURCES_BY_ID["adenocarcinoma-package"], max_chars=1400)
    assert chunks, "expected chunks"
    for c in chunks:
        assert c.text.startswith(f"Page: {page.title}\nPage type: Treatment package\nSection: ")
        assert c.url == SOURCES_BY_ID["adenocarcinoma-package"].url
    table_chunk = next(c for c in chunks if "$30,000" in c.text)
    assert "Cost of Adenocarcinoma Cancer Treatment" in table_chunk.section
    assert len({c.id for c in chunks}) == len(chunks)


def test_long_table_is_split_with_repeated_header():
    rows = "\n".join(f"| Treatment {i} | ${1000 + i} |" for i in range(80))
    page = ExtractedPage(url="u", title="Prices", sections=[Section(["Prices", "Price table"], "| Procedure | Price |\n| --- | --- |\n" + rows)])
    chunks = chunk_page(page, SOURCES_BY_ID["itc-price-list"], max_chars=600)
    assert len(chunks) > 1
    for c in chunks:
        assert "| Procedure | Price |" in c.text
    all_text = "\n".join(c.text for c in chunks)
    assert all(f"| Treatment {i} |" in all_text for i in range(80))
