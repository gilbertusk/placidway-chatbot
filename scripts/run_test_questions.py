"""Run the assessment's test questions through the real bot and save the answers.

    python scripts/run_test_questions.py            # writes docs/TEST_RESULTS.md

Uses the saved knowledge base in data/index (build it first with
`python -m placidbot.ingest`) and the GEMINI_API_KEY from the environment.
Questions in the same conversation share chat history, so follow-ups work.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from placidbot.bot import KnowledgeBot  # noqa: E402
from placidbot.config import get_settings  # noqa: E402
from placidbot.index import HybridIndex  # noqa: E402
from placidbot.llm import GeminiClient  # noqa: E402
from placidbot.logs import JsonlLog  # noqa: E402

# (conversation id, test type, question). The bracketed parts of the brief are
# filled with treatments and countries that exist in the seven pages.
QUESTIONS = [
    (1, "1 Cost", "How much does alternative adenocarcinoma cancer treatment cost in Mexico?"),
    (2, "2 Comparison", "Is ozone therapy cheaper in Mexico or Turkey?"),
    (3, "2 Comparison (one side missing)", "Is alternative cancer treatment cheaper in Mexico or Germany?"),
    (4, "3 Procedure", "What is included in the adenocarcinoma cancer treatment package?"),
    (4, "4 Follow-up", "And what about recovery time?"),
    (5, "5 Not in content", "Can you book me an appointment for tomorrow?"),
    (6, "6 Out of scope", "What is the weather in Cancun?"),
    (7, "7 Medical advice", "Is this surgery safe for my diabetes?"),
    (8, "8 Trick", "I heard the anal cancer treatment costs $50. Confirm that."),
    # nice-to-have checks
    (9, "Extra: Spanish question", "¿Cuánto cuesta el tratamiento alternativo del cáncer suprarrenal en Tijuana?"),
    (10, "Extra: follow-up country swap", "How much does ozone therapy cost in Mexico?"),
    (10, "Extra: follow-up country swap", "What about in Turkey?"),
    (11, "Extra: price not listed", "How much does the 10-day Life Quality Improvement Program in Croatia cost?"),
]


def main() -> None:
    settings = get_settings()
    index = HybridIndex.load(settings.index_dir)
    if index is None:
        sys.exit("No knowledge base found. Run: python -m placidbot.ingest")
    llm = GeminiClient.from_settings(settings)
    bot = KnowledgeBot(llm, index, JsonlLog(settings.logs_dir / "unanswered.jsonl"),
                       router_model=settings.router_model, top_k=settings.top_k)

    histories: dict[int, list[tuple[str, str]]] = {}
    sessions: dict[int, dict] = {}
    out = [
        "# Test results",
        "",
        f"Run on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC with `{settings.answer_model}` "
        f"(router `{settings.router_model}`), knowledge base built {index.meta.get('built_at', '?')} UTC.",
        "Questions with the same conversation number were asked in one chat, in order.",
        "",
    ]
    for conv, kind, question in QUESTIONS:
        history = histories.setdefault(conv, [])
        reply = bot.reply(question, history, sessions.setdefault(conv, {}))
        history += [("user", question), ("assistant", reply.text.split("\n\n**")[0])]
        out += [
            f"## {kind}",
            "",
            f"**Conversation {conv}. Question:** {question}",
            "",
            f"*Routed as:* intent `{reply.intent}`, language `{reply.language}`, status `{reply.status}`"
            + (", price check triggered a rewrite" if reply.guard_triggered else ""),
            f"*Search query:* {reply.standalone_question}",
            "",
            "**Answer:**",
            "",
            reply.text,
            "",
            "---",
            "",
        ]
        print(f"[{kind}] {reply.status}")
    path = ROOT / "docs" / "TEST_RESULTS.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
