"""Hybrid retrieval: BM25 keyword search + embedding similarity.

Why both:
* BM25 is exact. It reliably finds names and numbers ("adrenal", "ozone",
  "Out-Patient") that embeddings can blur.
* Embeddings find passages that use different words than the question
  ("how long will I stay" vs "3-week program") and work across languages.

Scores from each method are scaled to 0..1 and averaged. If embeddings are
unavailable (no API key, API down) the index silently runs on BM25 alone.

Two small boosts are added on top:
* priced passages, for cost questions;
* section headings that answer the question. Every chunk of a page starts
  with the page title, so for "what is included in the adenocarcinoma
  package?" all 22 chunks of that page match "adenocarcinoma ... package"
  and score almost the same. The one word that picks the right chunk is
  "included", and it is in the heading "Included in the Out-Patient Program".
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .chunker import Chunk
from .llm import LLMClient, LLMError

log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9]+")
HYPHENATED_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it me my of on or the to what which "
    "with you your this that there their about will would should could much many".split()
)
COST_WORDS = frozenset("cost costs price prices priced pricing cheap cheaper cheapest expensive how much fee fees usd".split())
MONEY_RE = re.compile(r"(\$|€|£|\bUSD\b|\bEUR\b|\bGBP\b)\s?\d|\d[\d,.]*\s?(USD|EUR|GBP)\b")
MONEY_BOOST = 0.1  # nudge priced passages up for cost questions
SECTION_MATCH_BOOST = 0.25  # enough to lift the matching section above its look-alike neighbours on the same page


def tokenize(text: str) -> list[str]:
    """Words minus stopwords, plus each hyphenated word joined up.

    Without the joined form "in-patient" is just "patient" ("in" is a
    stopword), the same as "out-patient", and an in-patient question would
    match the out-patient section as well as its own.
    """
    text = text.lower()
    tokens = [t for t in TOKEN_RE.findall(text) if t not in STOPWORDS]
    tokens += ["".join(TOKEN_RE.findall(word)) for word in HYPHENATED_RE.findall(text)]
    return tokens


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.zeros_like(values) if high <= 0 else np.ones_like(values)
    return (values - low) / (high - low)


@dataclass
class Hit:
    chunk: Chunk
    score: float


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray | None = None, meta: dict | None = None):
        self.chunks = chunks
        self.embeddings = embeddings
        self.meta = meta or {}
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None
        self._has_money = np.array([bool(MONEY_RE.search(c.text)) for c in chunks], dtype=bool)
        self._title_tokens = [frozenset(tokenize(c.page_title)) for c in chunks]
        self._section_tokens = [frozenset(tokenize(c.section)) for c in chunks]

    def __len__(self) -> int:
        return len(self.chunks)

    def section_match(self, query_tokens: list[str]) -> np.ndarray:
        """Per chunk, 0..1: of the question's words that the page title does not
        already cover, the share found in the chunk's section heading.

        Words in the page title say which page is meant; they cannot tell the
        page's chunks apart, so they are left out. A chunk on another page
        scores lower because more of the question is left unexplained.
        """
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        wanted = frozenset(query_tokens)
        for i, (title_tokens, section_tokens) in enumerate(zip(self._title_tokens, self._section_tokens)):
            specific = wanted - title_tokens
            if specific:
                scores[i] = len(specific & section_tokens) / len(specific)
        return scores

    def search(self, query: str, llm: LLMClient | None = None, k: int = 6, alpha: float = 0.5) -> list[Hit]:
        if not self.chunks:
            return []
        query_tokens = tokenize(query)
        keyword = np.asarray(self._bm25.get_scores(query_tokens), dtype=np.float32) if query_tokens else np.zeros(len(self.chunks))
        combined = _minmax(keyword)

        if self.embeddings is not None and llm is not None:
            try:
                q = llm.embed([query], task="RETRIEVAL_QUERY")[0]
                semantic = _minmax(self.embeddings @ q)
                combined = alpha * semantic + (1 - alpha) * combined
            except (LLMError, ValueError) as exc:  # API down or vector size mismatch: keyword search only
                log.warning("Query embedding failed, using BM25 only: %s", exc)

        if COST_WORDS & set(TOKEN_RE.findall(query.lower())) or "how much" in query.lower():
            combined = combined + MONEY_BOOST * self._has_money
        combined = combined + SECTION_MATCH_BOOST * self.section_match(query_tokens)

        order = np.argsort(-combined)[:k]
        return [Hit(chunk=self.chunks[i], score=float(combined[i])) for i in order if combined[i] > 0]

    # -- persistence ---------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        emb_path = directory / "embeddings.npy"
        if self.embeddings is not None:
            np.save(emb_path, self.embeddings)
        elif emb_path.exists():
            emb_path.unlink()

    @classmethod
    def load(cls, directory: Path) -> "HybridIndex | None":
        chunks_path = directory / "chunks.json"
        if not chunks_path.exists():
            return None
        chunks = [Chunk(**c) for c in json.loads(chunks_path.read_text(encoding="utf-8"))]
        emb_path = directory / "embeddings.npy"
        embeddings = np.load(emb_path) if emb_path.exists() else None
        if embeddings is not None and len(embeddings) != len(chunks):
            log.warning("Embeddings do not match chunks; ignoring them")
            embeddings = None
        meta_path = directory / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(chunks, embeddings, meta)
