from conftest import fixture_html

from placidbot.chunker import chunk_page
from placidbot.extract import ExtractedPage, Section, collapse_card_lists, drop_cross_page_boilerplate, extract_page
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


def _video_card(i: int) -> str:
    return (
        f'<div class="card"><h3 class="title"><a href="/video/{i}">Patient story number {i} at the clinic</a></h3>'
        "<p>Cancer Treatment</p><h4>ITC Immunity Therapy Center</h4><p>Tijuana, Mexico</p></div>"
    )


def test_card_grid_becomes_one_list_instead_of_many_tiny_sections():
    html = (
        "<html><body><main><h1>Clinic Videos</h1><h2>About the clinic</h2>"
        "<p>" + "The clinic offers individual programs for visiting patients. " * 6 + "</p>"
        "<h2>Videos</h2>" + "".join(_video_card(i) for i in range(8))
        + "<h2>Frequently asked questions</h2><p>" + "Patients usually stay three weeks in Tijuana. " * 5 + "</p>"
        "</main></body></html>"
    )
    page = extract_page(html, "https://example.com/videos")
    sections = collapse_card_lists(page.sections, page.title)

    listing = [s for s in sections if "Patient story number 0" in s.text]
    assert len(listing) == 1
    assert listing[0].heading_path == ["Clinic Videos", "Videos"]
    lines = listing[0].text.splitlines()
    assert len(lines) == 8
    assert lines[3] == "- Patient story number 3 at the clinic (Cancer Treatment; ITC Immunity Therapy Center; Tijuana, Mexico)"
    assert len(sections) == 3  # about + one list + FAQ, instead of 18


def test_short_runs_of_small_sections_are_left_alone():
    sections = [Section(["Page", f"Heading {i}"], "Short text.") for i in range(3)]
    assert collapse_card_lists(sections, "Page") == sections


def test_long_list_without_blank_lines_is_split_on_line_boundaries():
    lines = [f"- Video number {i} about a patient journey at the clinic" for i in range(100)]
    page = ExtractedPage(url="u", title="Videos", sections=[Section(["Videos", "All videos"], "\n".join(lines))])
    chunks = chunk_page(page, SOURCES_BY_ID["itc-videos"], max_chars=600)

    bodies = [c.text.split("\n\n", 1)[1] for c in chunks]
    assert len(chunks) > 5
    assert all(len(body) <= 600 for body in bodies)
    assert [ln for body in bodies for ln in body.splitlines()] == lines  # nothing lost, no line cut in half
