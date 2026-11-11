"""Measure retrieval quality on the real knowledge base.

    python scripts/eval_retrieval.py            # hybrid search (needs GEMINI_API_KEY)
    python scripts/eval_retrieval.py --bm25     # keyword search only, no API calls

Each case is a question (already standalone English, as the router would
produce) and the passage that must be retrieved for a correct answer: the
page it is on and a phrase it contains. The script reports the rank of that
passage and how many cases have it inside the top K given to the answer
model. Run it before and after changing chunking or ranking.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from placidbot.config import get_settings  # noqa: E402
from placidbot.index import HybridIndex  # noqa: E402
from placidbot.llm import GeminiClient  # noqa: E402


@dataclass(frozen=True)
class Case:
    question: str
    source_id: str | None  # None = any page
    needle: str  # text the right passage contains


CASES = (
    Case("What is included in the alternative adenocarcinoma cancer treatment package at ITC in Tijuana?",
         "adenocarcinoma-package", "Section: Included in the Out-Patient Program"),
    Case("What is included in the alternative adenocarcinoma cancer treatment package at ITC in Tijuana?",
         "adenocarcinoma-package", "Section: Included in the In-Patient Program"),
    Case("What is included in the in-patient program for alternative adrenal cancer treatment at ITC?",
         "adrenal-package", "Section: Included in the In-Patient Program"),
    Case("What is not included in the alternative anal cancer treatment package at ITC in Tijuana?",
         "anal-package", "Section: Excluded Services"),
    Case("How much does alternative adrenal cancer treatment cost in Tijuana?",
         "adrenal-package", "| **Out-Patient Program** | 3 Weeks | $18,995 |"),
    Case("How much does alternative adenocarcinoma cancer treatment cost in Mexico?",
         "adenocarcinoma-package", "| **In-Patient Program** | 3 Weeks | $30,000 |"),
    Case("What optional procedures are available for adenocarcinoma treatment at ITC and how much do they cost?",
         "adenocarcinoma-package", "Intratumoral Immunotherapy"),
    Case("Why is the treatment period three weeks for alternative adenocarcinoma cancer treatment at ITC?",
         "adenocarcinoma-package", "5. Why is the treatment period three weeks?"),
    Case("What does the typical 3-week itinerary look like for alternative anal cancer treatment at ITC?",
         "anal-package", "Section: 3-Week Typical Itinerary"),
    Case("Who should consider alternative adrenal cancer treatment at Immunity Therapy Center?",
         "adrenal-package", "Section: Who Should Consider"),
    Case("How much does ozone therapy cost in Turkey?", "alt-medicine-prices", "Fizikon Medical Center"),
    Case("How much does ozone therapy cost in Mexico?", "alt-medicine-prices", "Stem Health Guadalajara"),
    Case("How much does the 10-day Life Quality Improvement Program in Croatia cost?",
         "alt-medicine-prices", "Life Quality Improvement Program"),
    Case("Which ITC videos are about Lyme disease treatment?", "itc-videos", "Lyme Disease"),
    Case("Who founded ITC Immunity Therapy Center?", None, "Carlos Bautista"),
    Case("How much does breast cancer treatment cost at ITC Immunity Therapy Center?",
         "itc-price-list", "| Breast Cancer | Breast Cancer | $18995 |"),
)


def rank_of(case: Case, index: HybridIndex, llm: GeminiClient | None) -> int | None:
    hits = index.search(case.question, llm, k=len(index))
    for position, hit in enumerate(hits, start=1):
        if case.source_id in (None, hit.chunk.source_id) and case.needle in hit.chunk.text:
            return position
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Report where the expected passage ranks for each test question.")
    parser.add_argument("--bm25", action="store_true", help="keyword search only (no API key needed)")
    args = parser.parse_args()

    settings = get_settings()
    index = HybridIndex.load(settings.index_dir)
    if index is None:
        sys.exit("No knowledge base found. Run: python -m placidbot.ingest")
    llm = None
    if not args.bm25:
        llm = GeminiClient.from_settings(settings)

    found = 0
    for case in CASES:
        rank = rank_of(case, index, llm)
        hit = rank is not None and rank <= settings.top_k
        found += hit
        print(f"{'ok  ' if hit else 'MISS'} rank {rank if rank is not None else '-':>3}  {case.question[:78]}  [{case.needle[:40]}]")
    mode = "BM25 only" if llm is None else "hybrid"
    print(f"\n{found}/{len(CASES)} expected passages inside the top {settings.top_k} ({mode}, {len(index)} passages).")


if __name__ == "__main__":
    main()
