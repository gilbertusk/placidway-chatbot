import pytest
from conftest import FakeLLM, fixture_html

from placidbot.bot import DISCLAIMER, FALLBACK, UNAVAILABLE, KnowledgeBot
from placidbot.chunker import chunk_page
from placidbot.extract import extract_page
from placidbot.index import HybridIndex
from placidbot.llm import LLMError
from placidbot.logs import JsonlLog
from placidbot.sources import SOURCES_BY_ID

PACKAGE_URL = SOURCES_BY_ID["adenocarcinoma-package"].url


@pytest.fixture
def index():
    chunks = []
    for name, sid in [("package.html", "adenocarcinoma-package"), ("price_list.html", "itc-price-list"), ("comparison.html", "alt-medicine-prices")]:
        src = SOURCES_BY_ID[sid]
        chunks += chunk_page(extract_page(fixture_html(name), src.url), src)
    llm = FakeLLM()
    return HybridIndex(chunks, llm.embed([c.text for c in chunks], task="RETRIEVAL_DOCUMENT"))


def label_of(prompt: str, needle: str) -> str:
    """Find the [S#] label of the passage that contains `needle`."""
    for block in prompt.split("\n\n["):
        if needle in block:
            return block.split("]", 1)[0].lstrip("[")
    raise AssertionError(f"{needle!r} not retrieved")


def make_bot(llm, index, tmp_path):
    return KnowledgeBot(llm, index, JsonlLog(tmp_path / "unanswered.jsonl"))


def test_grounded_answer_gets_source_link(index, tmp_path):
    llm = FakeLLM(answer_fn=lambda p: {
        "status": "answered", "answer": "The out-patient program is $18,995 (price starting from), 3 weeks.",
        "source_ids": [label_of(p, "$30,000")], "medical_topic": False})
    reply = make_bot(llm, index, tmp_path).reply("How much does adenocarcinoma treatment cost in Mexico?", [])
    assert reply.status == "answered"
    assert f"]({PACKAGE_URL})" in reply.text
    assert "**Sources:**" in reply.text
    assert not reply.guard_triggered


def test_invented_price_is_rewritten(index, tmp_path):
    drafts = iter(["It costs about $12,000.", "It is listed from $18,995."])
    llm = FakeLLM(answer_fn=lambda p: {
        "status": "answered", "answer": next(drafts), "source_ids": [label_of(p, "$30,000")], "medical_topic": False})
    reply = make_bot(llm, index, tmp_path).reply("How much is adenocarcinoma treatment?", [])
    assert reply.guard_triggered
    assert "$18,995" in reply.text and "12,000" not in reply.text
    assert "CORRECTION REQUIRED" in llm.answer_prompts[1] and "12000" in llm.answer_prompts[1]


def test_persistent_invention_falls_back_and_is_logged(index, tmp_path):
    llm = FakeLLM(answer_fn=lambda p: {
        "status": "answered", "answer": "It costs $9,999.", "source_ids": ["S1"], "medical_topic": False})
    bot = make_bot(llm, index, tmp_path)
    reply = bot.reply("How much is adenocarcinoma treatment?", [])
    assert reply.status == "not_found"
    assert reply.text.startswith(FALLBACK["en"])
    assert "9,999" not in reply.text
    logged = bot.unanswered_log.read()
    assert logged and logged[0]["guard_triggered"] is True


def test_answer_without_valid_citation_is_rejected(index, tmp_path):
    llm = FakeLLM(answers=[
        {"status": "answered", "answer": "ITC is in Tijuana.", "source_ids": ["S99"], "medical_topic": False},
        {"status": "answered", "answer": "ITC is in Tijuana.", "source_ids": [], "medical_topic": False},
    ])
    reply = make_bot(llm, index, tmp_path).reply("Where is ITC?", [])
    assert reply.status == "not_found"
    assert reply.sources == []


def test_not_found_is_logged_as_content_gap(index, tmp_path):
    llm = FakeLLM(answers=[{"status": "not_found", "answer": "I don't have that information.", "source_ids": [], "medical_topic": False}])
    bot = make_bot(llm, index, tmp_path)
    bot.reply("Do you have hip replacement in Spain?", [])
    assert bot.unanswered_log.read()[0]["question"] == "Do you have hip replacement in Spain?"


def test_out_of_scope_skips_retrieval(index, tmp_path):
    llm = FakeLLM(route={"standalone_question": "What is the weather in Cancun?", "language": "en", "intent": "out_of_scope"},
                  answers=[{"status": "info", "answer": "I can only help with PlacidWay questions.", "source_ids": [], "medical_topic": False}])
    reply = make_bot(llm, index, tmp_path).reply("What is the weather in Cancun?", [])
    assert "PASSAGES: (none)" in llm.answer_prompts[0]
    assert reply.status == "info" and reply.sources == []


def test_medical_disclaimer_shown_once_per_session(index, tmp_path):
    answer = {"status": "info", "answer": "I can't give medical advice; please ask your doctor.", "source_ids": [], "medical_topic": True}
    llm = FakeLLM(route={"standalone_question": "Is this safe with diabetes?", "language": "en", "intent": "medical_advice"},
                  answers=[dict(answer), dict(answer)])
    bot = make_bot(llm, index, tmp_path)
    session = {}
    first = bot.reply("Is this surgery safe for my diabetes?", [], session)
    second = bot.reply("And for my blood pressure?", [], session)
    assert DISCLAIMER["en"] in first.text
    assert DISCLAIMER["en"] not in second.text


def test_router_question_drives_retrieval(index, tmp_path):
    llm = FakeLLM(route={"standalone_question": "What is the duration of the adenocarcinoma in-patient program?", "language": "es", "intent": "question"},
                  answers=[{"status": "answered", "answer": "3 semanas.", "source_ids": ["S1"], "medical_topic": False}])
    make_bot(llm, index, tmp_path).reply("¿Y cuánto dura?", [("user", "adenocarcinoma?"), ("assistant", "...")])
    prompt = llm.answer_prompts[0]
    assert "In-Patient Program" in prompt
    assert "REPLY LANGUAGE: es" in prompt


def test_llm_outage_returns_friendly_error(index, tmp_path):
    class Broken(FakeLLM):
        def generate_json(self, **kwargs):
            if "standalone_question" in kwargs["schema"]["properties"]:
                return super().generate_json(**kwargs)
            raise LLMError("503")

    reply = make_bot(Broken(), index, tmp_path).reply("price?", [])
    assert reply.text == UNAVAILABLE and reply.status == "error"


def test_no_index_means_unavailable(tmp_path):
    reply = KnowledgeBot(FakeLLM(), None).reply("hi", [])
    assert reply.status == "error"
