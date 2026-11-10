"""Split extracted pages into retrieval chunks.

Rules:
* A chunk never crosses a section boundary, so a price table stays next to
  the heading that says what the price is for.
* Every chunk starts with a small header (page title, page type, section
  path). Retrieval and the LLM then always know which page and section a
  passage came from, even when the passage itself only says "$18,995".
* Long sections are split on blank lines; long tables are split by rows
  and each piece repeats the table's header row.
* Tiny sections (a heading with one short line) are merged into the next
  section of the same page so they are not lost or retrieved alone.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from .extract import ExtractedPage, Section
from .sources import Source


@dataclass
class Chunk:
    id: str
    source_id: str
    url: str
    page_title: str
    page_type: str
    section: str
    text: str  # header + body: what is embedded, searched and shown to the LLM

    def to_dict(self) -> dict:
        return asdict(self)


def _split_table(table_lines: list[str], max_chars: int) -> list[str]:
    header = table_lines[:2] if len(table_lines) >= 2 and set(table_lines[1].replace("|", "").strip()) <= set("-: ") else table_lines[:1]
    rows = table_lines[len(header):]
    pieces, current = [], list(header)
    for row in rows:
        if len("\n".join(current + [row])) > max_chars and len(current) > len(header):
            pieces.append("\n".join(current))
            current = list(header)
        current.append(row)
    if len(current) > len(header) or not pieces:
        pieces.append("\n".join(current))
    return pieces


def _blocks(text: str) -> list[str]:
    """Paragraph-level blocks; a table is one block."""
    blocks, current, in_table = [], [], False
    for line in text.splitlines():
        is_table = line.strip().startswith("|")
        if not line.strip():
            if current:
                blocks.append("\n".join(current))
                current, in_table = [], False
            continue
        if current and is_table != in_table:
            blocks.append("\n".join(current))
            current = []
        current.append(line)
        in_table = is_table
    if current:
        blocks.append("\n".join(current))
    return blocks


def _split_body(body: str, max_chars: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    pieces: list[str] = []
    current = ""
    for block in _blocks(body):
        if block.startswith("|") and len(block) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_split_table(block.splitlines(), max_chars))
            continue
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = block
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    prefix = []
    for x, y in zip(a, b):
        if x != y:
            break
        prefix.append(x)
    return prefix


def _join(first: Section, second: Section) -> Section:
    """Merge two sections; each keeps its own heading as a line in the text."""
    parent = _common_prefix(first.heading_path, second.heading_path)

    def labelled(sec: Section) -> str:
        own = sec.heading_path[len(parent):]
        return (" > ".join(own) + ":\n" + sec.text) if own else sec.text

    return Section(heading_path=parent, text=f"{labelled(first)}\n\n{labelled(second)}")


def _merge_tiny_sections(sections: list[Section], min_chars: int = 120) -> list[Section]:
    """Fold sections shorter than `min_chars` into the following section."""
    merged: list[Section] = []
    carry: Section | None = None
    for sec in sections:
        if carry is not None:
            sec = _join(carry, sec)
            carry = None
        if len(sec.text) < min_chars:
            carry = sec
            continue
        merged.append(sec)
    if carry is not None:
        if merged:
            merged[-1] = _join(merged[-1], carry)
        else:
            merged.append(carry)
    return merged


def chunk_page(page: ExtractedPage, source: Source, max_chars: int = 1400) -> list[Chunk]:
    sections = list(page.sections)
    if page.structured_prices:
        sections.insert(0, Section(
            heading_path=["Structured price data published on the page"],
            text="\n".join(f"- {p}" for p in page.structured_prices),
        ))
    sections = _merge_tiny_sections(sections)

    chunks: list[Chunk] = []
    seen_bodies: set[str] = set()
    for sec in sections:
        # the page title is already in the header, so leave it out of the section path
        section_label = " > ".join(h for h in sec.heading_path if h != page.title) or "Overview"
        for body in _split_body(sec.text, max_chars):
            digest = hashlib.sha1(body.encode()).hexdigest()
            if digest in seen_bodies:
                continue
            seen_bodies.add(digest)
            header = f"Page: {page.title}\nPage type: {source.page_type}\nSection: {section_label}"
            chunks.append(Chunk(
                id=f"{source.id}#{len(chunks) + 1}",
                source_id=source.id,
                url=source.url,
                page_title=page.title,
                page_type=source.page_type,
                section=section_label,
                text=f"{header}\n\n{body}",
            ))
    return chunks
