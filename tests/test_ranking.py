"""Ranking inside one page: the section heading decides between look-alike chunks."""

import numpy as np

from placidbot.chunker import Chunk
from placidbot.index import HybridIndex, tokenize

PAGE_TITLE = "Alternative Adenocarcinoma Cancer Treatment Package in Tijuana, Mexico by ITC"
FILLER = (
    "The alternative adenocarcinoma cancer treatment package at ITC in Tijuana is explained here. "
    "Meals are included and support is included for every adenocarcinoma cancer treatment package."
)
# As on the real page: a long list that also uses the page's own words. Its length puts it
# slightly below the short FAQ chunks in plain BM25: a near-tie the heading has to break.
INCLUSIONS = (
    "- **Medications:** standard medications included in the adenocarcinoma cancer treatment package at ITC in Tijuana\n"
    + "\n".join(f"- **{item}:** scheduled by the doctors" for item in ("Weekly lab work", "Consultations", "IV therapies"))
)
UNRELATED = "Patient stories and interviews recorded at the clinic, grouped by condition and by year of recording."
PRICE_TABLE = "| Program | Duration | Price |\n| --- | --- | --- |\n| Out-Patient Program | 3 Weeks | $18,995 |\n" + INCLUSIONS


def _chunk(number: int, section: str, body: str, title: str = PAGE_TITLE, source_id: str = "adenocarcinoma-package") -> Chunk:
    header = f"Page: {title}\nPage type: Treatment package\nSection: {section}"
    return Chunk(id=f"{source_id}#{number}", source_id=source_id, url="https://example.com/p", page_title=title,
                 page_type="Treatment package", section=section, text=f"{header}\n\n{body}")


def _page_index() -> HybridIndex:
    """Like the real package pages: every chunk repeats the page's own words, so
    only the section heading tells which chunk answers "what is included?"."""
    chunks = [_chunk(i, f"FAQs > {i}. A question about safety, staff or meals", FILLER) for i in range(1, 9)]
    chunks.append(_chunk(9, "Included in the Out-Patient Program ($18,995 USD)", INCLUSIONS))
    chunks.append(_chunk(10, "Price of Adenocarcinoma Cancer Treatment in Tijuana, Mexico", PRICE_TABLE))
    chunks += [_chunk(i, "Patient stories", UNRELATED, title="Clinic Videos", source_id="itc-videos") for i in (11, 12)]
    return HybridIndex(chunks)


def test_section_heading_breaks_the_near_tie_between_chunks_of_one_page():
    index = _page_index()
    question = "What is included in the alternative adenocarcinoma cancer treatment package at ITC in Tijuana?"

    keyword_best = index.chunks[int(np.argmax(index._bm25.get_scores(tokenize(question))))]
    assert keyword_best.section.startswith("FAQs")  # on keywords alone a FAQ chunk wins

    hits = index.search(question, k=3)
    assert hits[0].chunk.section.startswith("Included in the Out-Patient Program")


def test_price_question_ranks_the_price_section_first():
    hits = _page_index().search("What is the price of alternative adenocarcinoma cancer treatment in Mexico?", k=3)
    assert hits[0].chunk.section.startswith("Price of Adenocarcinoma")


def test_section_match_ignores_words_the_page_title_already_covers():
    index = _page_index()
    match = index.section_match(tokenize("What is included in the adenocarcinoma cancer treatment package?"))
    assert match[8] == 1.0  # "included" is the only word the title does not explain, and the heading has it
    assert match[0] == 0.0  # FAQ headings do not mention it
    assert match[9] == 0.0  # the price section does not either


def test_section_match_is_partial_on_a_page_about_something_else():
    other = _chunk(1, "Included in the Out-Patient Program", "- Weekly lab work",
                   title="Alternative Adrenal Cancer Treatment Package in Tijuana, Mexico by ITC", source_id="adrenal-package")
    index = HybridIndex([other])
    match = index.section_match(tokenize("What is included in the adenocarcinoma cancer treatment package?"))
    assert match[0] == 0.5  # "included" matches, "adenocarcinoma" does not: this is the wrong page
