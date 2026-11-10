"""The chat pipeline for one user message.

    message + history
      -> 1. route      (LLM: standalone English question, language, intent)
      -> 2. retrieve   (hybrid search over the knowledge base)
      -> 3. answer     (LLM: reply from passages only, JSON with cited ids)
      -> 4. verify     (code: cited ids exist, every money amount is in the cited text)
             failed once -> ask the model to rewrite; failed twice -> safe "I don't know"
      -> 5. compose    (append source links, one-time medical disclaimer)
      -> 6. log        (content gaps go to the unanswered log)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .guard import unsupported_amounts
from .index import HybridIndex
from .llm import LLMClient, LLMError
from .logs import JsonlLog
from .prompts import (
    ANSWER_SCHEMA, ANSWER_SYSTEM, INTENTS, ROUTER_SCHEMA, ROUTER_SYSTEM, STATUSES, answer_prompt, router_prompt,
)

log = logging.getLogger(__name__)

NO_RETRIEVAL_INTENTS = {"booking", "out_of_scope", "greeting"}
HISTORY_TURNS = 6  # messages of history given to the router

DISCLAIMER = {
    "en": "This information comes from PlacidWay's website and is not medical advice. Please discuss any treatment with your doctor.",
    "es": "Esta información proviene del sitio web de PlacidWay y no es consejo médico. Consulte cualquier tratamiento con su médico.",
    "ar": "هذه المعلومات مأخوذة من موقع PlacidWay وليست نصيحة طبية. يرجى مناقشة أي علاج مع طبيبك.",
    "id": "Informasi ini berasal dari situs PlacidWay dan bukan nasihat medis. Diskusikan setiap pengobatan dengan dokter Anda.",
}
FALLBACK = {
    "en": "I'm sorry, I don't have reliable information on that in PlacidWay's pages. "
          "You can request a free quote from PlacidWay using the form below this chat, and their team will answer you directly.",
    "es": "Lo siento, no tengo información fiable sobre eso en las páginas de PlacidWay. "
          "Puede solicitar una cotización gratuita a PlacidWay con el formulario debajo de este chat.",
    "ar": "عذرًا، لا تتوفر لدي معلومات موثوقة عن ذلك في صفحات PlacidWay. يمكنك طلب عرض سعر مجاني من PlacidWay عبر النموذج أسفل هذه المحادثة.",
    "id": "Maaf, saya tidak punya informasi yang dapat diandalkan tentang itu di halaman PlacidWay. "
          "Anda bisa meminta penawaran gratis dari PlacidWay melalui formulir di bawah chat ini.",
}
UNAVAILABLE = (
    "Sorry, I can't reach the language model right now. Please try again in a minute, "
    "or request a free quote from PlacidWay using the form below this chat."
)
SOURCES_LABEL = {"en": "Sources", "es": "Fuentes", "ar": "المصادر", "id": "Sumber"}


@dataclass
class Route:
    standalone_question: str
    language: str
    intent: str


@dataclass
class BotReply:
    text: str
    status: str
    intent: str
    language: str
    standalone_question: str
    sources: list[tuple[str, str]] = field(default_factory=list)  # (title, url)
    retrieved_ids: list[str] = field(default_factory=list)
    guard_triggered: bool = False


def _pick(table: dict[str, str], language: str) -> str:
    return table.get((language or "en").lower()[:2], table["en"])


class KnowledgeBot:
    def __init__(self, llm: LLMClient | None, index: HybridIndex | None, unanswered_log: JsonlLog | None = None,
                 router_model: str | None = None, top_k: int = 6):
        self.llm = llm
        self.index = index
        self.unanswered_log = unanswered_log
        self.router_model = router_model
        self.top_k = top_k

    # 1. route ------------------------------------------------------------
    def route(self, message: str, history: list[tuple[str, str]]) -> Route:
        try:
            data = self.llm.generate_json(
                system=ROUTER_SYSTEM,
                prompt=router_prompt(history=history[-HISTORY_TURNS:], message=message),
                schema=ROUTER_SCHEMA,
                model=self.router_model,
            )
            intent = data.get("intent") if data.get("intent") in INTENTS else "question"
            question = (data.get("standalone_question") or message).strip()
            language = (data.get("language") or "en").strip().lower()[:5] or "en"
            return Route(question, language, intent)
        except (LLMError, AttributeError, TypeError) as exc:
            log.warning("Router failed, searching with the raw message: %s", exc)
            return Route(message, "en", "question")

    # 3. answer ---------------------------------------------------------
    def _generate(self, message: str, route: Route, labelled: list[tuple[str, str]], correction: str = "") -> dict:
        data = self.llm.generate_json(
            system=ANSWER_SYSTEM,
            prompt=answer_prompt(
                user_message=message, standalone_question=route.standalone_question, intent=route.intent,
                language=route.language, passages=labelled, correction=correction,
            ),
            schema=ANSWER_SCHEMA,
        )
        if data.get("status") not in STATUSES or not isinstance(data.get("answer"), str):
            raise LLMError(f"Answer JSON has the wrong shape: {data!r}"[:300])
        return data

    # 4. verify -------------------------------------------------------------
    @staticmethod
    def _verify(data: dict, message: str, by_label: dict) -> tuple[list[str], list[str]]:
        """Returns (valid cited labels, problems found)."""
        problems: list[str] = []
        cited = [sid for sid in dict.fromkeys(data.get("source_ids") or []) if sid in by_label]
        if data["status"] in ("answered", "partial") and not cited:
            problems.append("the answer states facts but cites no passage")
        cited_text = "\n".join(by_label[sid].text for sid in cited)
        bad = unsupported_amounts(data["answer"], cited_text, message)
        if bad:
            problems.append("these amounts do not appear in the cited passages: " + ", ".join(bad))
        return cited, problems

    # main entry -----------------------------------------------------------
    def reply(self, message: str, history: list[tuple[str, str]], session: dict | None = None) -> BotReply:
        session = session if session is not None else {}
        message = (message or "").strip()
        if self.llm is None or self.index is None or len(self.index) == 0:
            return BotReply(UNAVAILABLE, "error", "question", "en", message)

        try:
            route = self.route(message, history)

            hits = []
            if route.intent not in NO_RETRIEVAL_INTENTS:
                hits = self.index.search(route.standalone_question, self.llm, k=self.top_k)
            by_label = {f"S{i + 1}": h.chunk for i, h in enumerate(hits)}
            labelled = [(label, chunk.text) for label, chunk in by_label.items()]

            data = self._generate(message, route, labelled)
            cited, problems = self._verify(data, message, by_label)
            guard_triggered = False
            if problems:
                guard_triggered = True
                log.warning("Answer failed verification (%s); asking for a rewrite", "; ".join(problems))
                data = self._generate(message, route, labelled, correction=(
                    "Your previous draft failed verification: " + "; ".join(problems)
                    + ". Rewrite it using only facts and amounts that appear verbatim in the passages you cite,"
                      " or say the information is not listed."
                ))
                cited, problems = self._verify(data, message, by_label)
                if problems:
                    log.warning("Rewrite still failed verification (%s); using safe fallback", "; ".join(problems))
                    data = {"status": "not_found", "answer": _pick(FALLBACK, route.language), "source_ids": [], "medical_topic": False}
                    cited = []
        except LLMError as exc:
            log.error("LLM failure: %s", exc)
            return BotReply(UNAVAILABLE, "error", "question", "en", message)

        # 5. compose -------------------------------------------------------
        sources = list(dict.fromkeys((by_label[sid].page_title, by_label[sid].url) for sid in cited))
        text = data["answer"].strip()
        if sources:
            text += f"\n\n**{_pick(SOURCES_LABEL, route.language)}:**\n" + "\n".join(f"- [{t}]({u})" for t, u in sources)
        needs_disclaimer = bool(data.get("medical_topic")) or route.intent == "medical_advice"
        if needs_disclaimer and not session.get("disclaimer_shown"):
            text += f"\n\n_{_pick(DISCLAIMER, route.language)}_"
            session["disclaimer_shown"] = True

        reply = BotReply(
            text=text, status=data["status"], intent=route.intent, language=route.language,
            standalone_question=route.standalone_question, sources=sources,
            retrieved_ids=[c.id for c in by_label.values()], guard_triggered=guard_triggered,
        )

        # 6. log content gaps ---------------------------------------------
        if self.unanswered_log is not None and (reply.status in ("not_found", "partial") or guard_triggered):
            self.unanswered_log.append({
                "question": message, "standalone_question": route.standalone_question,
                "status": reply.status, "intent": route.intent, "language": route.language,
                "guard_triggered": guard_triggered, "retrieved": reply.retrieved_ids,
            })
        return reply
