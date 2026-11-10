"""Gradio chat interface.

Visitors get the chat and a quote form. The admin panel (refresh content,
see unanswered questions) is locked behind the ADMIN_TOKEN secret so that
random visitors cannot make the bot re-crawl PlacidWay's site.
"""

from __future__ import annotations

import hmac
import logging
import re
import threading

import gradio as gr

from .bot import UNAVAILABLE, KnowledgeBot
from .config import Settings
from .index import HybridIndex
from .ingest import build_index
from .llm import LLMClient
from .logs import JsonlLog
from .sources import SOURCES

log = logging.getLogger(__name__)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

EXAMPLES = [
    "How much does alternative adenocarcinoma cancer treatment cost in Mexico?",
    "What is included in the anal cancer treatment package?",
    "Is ozone therapy cheaper in Mexico or Turkey?",
    "¿Cuánto cuesta el tratamiento del cáncer suprarrenal en Tijuana?",
]

INTRO = """# PlacidWay Assistant
Ask about **ITC Immunity Therapy Center** (Tijuana, Mexico): its alternative cancer treatment packages,
prices and videos, and **alternative medicine prices abroad**. Answers come only from seven PlacidWay
pages and always link to the page they came from. If something isn't on those pages, the assistant says so.
"""


def _text_of(content) -> str:
    """Gradio message content can be a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content or "")


def _history_pairs(messages: list[dict]) -> list[tuple[str, str]]:
    pairs = []
    for m in messages or []:
        text = _text_of(m.get("content"))
        text = text.split("\n\n**")[0]  # drop the appended sources/disclaimer block
        pairs.append((m.get("role", "user"), text[:800]))
    return pairs


class App:
    def __init__(self, settings: Settings, llm: LLMClient | None, index: HybridIndex | None):
        self.settings = settings
        self.llm = llm
        self.unanswered = JsonlLog(settings.logs_dir / "unanswered.jsonl")
        self.leads = JsonlLog(settings.logs_dir / "leads.jsonl")
        self.bot = KnowledgeBot(llm, index, self.unanswered, router_model=settings.router_model, top_k=settings.top_k)
        self._refresh_lock = threading.Lock()

    # chat ---------------------------------------------------------------
    def respond(self, message: str, messages: list[dict], session: dict):
        message = (message or "").strip()
        if not message:
            return "", messages, session
        session = dict(session or {})
        try:
            text = self.bot.reply(message, _history_pairs(messages), session).text
        except Exception:  # never show a stack trace to a visitor
            log.exception("Unexpected error answering %r", message)
            text = UNAVAILABLE
        messages = (messages or []) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": text},
        ]
        return "", messages, session

    # quote form ---------------------------------------------------------
    def save_lead(self, name: str, email: str, note: str, consent: bool):
        name, email = (name or "").strip(), (email or "").strip()
        if not name or not EMAIL_RE.match(email):
            return "Please enter your name and a valid email address."
        if not consent:
            return "Please tick the box so PlacidWay may contact you about your quote."
        self.leads.append({"name": name, "email": email, "note": (note or "").strip()[:1000]})
        return f"Thank you, {name}. PlacidWay's team will contact you at {email} with a free quote."

    # admin --------------------------------------------------------------
    def _authorised(self, token: str) -> bool:
        return bool(self.settings.admin_token) and hmac.compare_digest((token or "").strip(), self.settings.admin_token)

    def refresh(self, token: str):
        if not self._authorised(token):
            return "Admin token missing or incorrect.", self.kb_status()
        if not self._refresh_lock.acquire(blocking=False):
            return "A refresh is already running.", self.kb_status()
        try:
            index, report = build_index(self.settings, self.llm)
            self.bot.index = index  # swap in the new knowledge base atomically
            return report.summary(), self.kb_status()
        except Exception as exc:  # keep serving the old index on any failure
            return f"Refresh failed, previous content kept: {exc}", self.kb_status()
        finally:
            self._refresh_lock.release()

    def show_unanswered(self, token: str):
        if not self._authorised(token):
            return [["Admin token missing or incorrect.", "", "", ""]]
        rows = [[r.get("time", ""), r.get("question", ""), r.get("status", ""), r.get("language", "")]
                for r in reversed(self.unanswered.read())]
        return rows or [["No unanswered questions yet.", "", "", ""]]

    def kb_status(self) -> str:
        index = self.bot.index
        if index is None or len(index) == 0:
            return "Knowledge base not built yet."
        meta = index.meta
        lines = [
            f"**Last refreshed:** {meta.get('built_at', 'unknown')} UTC · **{len(index)} passages** · "
            f"semantic search {'on' if index.embeddings is not None else 'off (keyword only)'}",
            "",
        ]
        lines += [f"- {s.page_type}: [{s.url.split('/')[-1].replace('-', ' ')}]({s.url})" for s in SOURCES]
        return "\n".join(lines)

    # layout -------------------------------------------------------------
    def build(self) -> gr.Blocks:
        with gr.Blocks(title="PlacidWay Assistant") as demo:
            session = gr.State({})
            gr.Markdown(INTRO)
            chatbot = gr.Chatbot(height=480, label="Chat", buttons=["copy"], feedback_options=None)
            with gr.Row():
                box = gr.Textbox(placeholder="Ask about treatments, packages or prices…", show_label=False, scale=8, autofocus=True)
                send = gr.Button("Send", variant="primary", scale=1)
            with gr.Row():
                clear = gr.Button("New conversation", size="sm")
            gr.Examples(EXAMPLES, inputs=box, label="Try asking")

            with gr.Accordion("Request a free quote", open=False):
                name = gr.Textbox(label="Name")
                email = gr.Textbox(label="Email")
                note = gr.Textbox(label="What treatment are you interested in?", lines=2)
                consent = gr.Checkbox(label="I agree that PlacidWay may contact me about my quote.")
                lead_btn = gr.Button("Send request")
                lead_msg = gr.Markdown()

            with gr.Accordion("Knowledge base", open=False):
                status = gr.Markdown(self.kb_status())

            with gr.Accordion("Admin", open=False):
                token = gr.Textbox(label="Admin token", type="password")
                with gr.Row():
                    refresh_btn = gr.Button("Refresh content from PlacidWay")
                    gaps_btn = gr.Button("Show unanswered questions")
                refresh_out = gr.Textbox(label="Refresh report", lines=4, interactive=False)
                gaps = gr.Dataframe(headers=["Time (UTC)", "Question", "Status", "Language"], interactive=False, wrap=True)

            box.submit(self.respond, [box, chatbot, session], [box, chatbot, session])
            send.click(self.respond, [box, chatbot, session], [box, chatbot, session])
            clear.click(lambda: ([], {}, ""), None, [chatbot, session, box])
            lead_btn.click(self.save_lead, [name, email, note, consent], lead_msg)
            refresh_btn.click(self.refresh, token, [refresh_out, status])
            gaps_btn.click(self.show_unanswered, token, gaps)
        return demo
