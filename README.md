---
title: PlacidWay Knowledge Chatbot
emoji: 💬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Chatbot for 7 PlacidWay pages with cited sources
---

# PlacidWay Knowledge Chatbot

A chatbot that answers visitor questions using only seven PlacidWay pages. Every answer links to the
page it came from, and when the pages do not contain the answer the bot says so instead of guessing.

- **Live demo:** https://huggingface.co/spaces/gilb3rttuss/placidway-chatbot
- **Test questions and answers:** [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md)

## Features

| Area | How it works |
|---|---|
| Content ingestion | `placidbot/crawler.py` fetches the 7 assigned URLs only (no link following), respects robots.txt, sends a descriptive User-Agent and waits 1 s between requests. |
| Knowledge base | Pages are cleaned to Markdown and split by heading into chunks. Tables stay whole; long tables are split with their header repeated. Search is hybrid BM25 + embeddings. |
| Grounded answers | The model only sees the retrieved passages and returns the ids of the ones it used. The app turns ids into source links, so the model never writes a URL. |
| "I don't know" | Enforced in code as well as in the prompt. An answer without a valid citation, or with a price that is not in the cited text, is sent back for one rewrite and otherwise replaced by a fixed "I don't have that information" reply with a link to the free quote. |
| Prices | Copied with currency and context ("starting from", program, duration). The model may not calculate, convert or estimate amounts. |
| Chat UI | Gradio, with example questions, a free-quote form and a knowledge-base panel. |
| Refresh | `python -m placidbot.ingest` or the admin "Refresh content" button. The report lists changed pages; a page that fails to fetch keeps its previous content. |
| Extras | Replies in the visitor's language, follow-up questions, a medical disclaimer once per conversation, lead capture, and a log of unanswered questions in the admin panel. |

## How a message is answered

```
user message + chat history
  │
  ▼
1. Route     small Gemini model → standalone English question, language, intent
  ▼
2. Retrieve  hybrid BM25 + embedding search, top 6 passages
             (skipped for booking, out-of-scope and greeting messages)
  ▼
3. Answer    Gemini → {status, answer, source_ids, medical_topic} as JSON
  ▼
4. Verify    cited ids exist? every money amount appears in the cited passages?
             fail → one rewrite → still failing → safe fallback reply
  ▼
5. Compose   answer + source links + disclaimer (first medical reply only)
  ▼
6. Log       not_found / partial / rewritten answers → data/logs/unanswered.jsonl
```

Ingestion runs at start-up when no index exists, from the CLI, or from the admin panel:

```
7 URLs → crawl → extract (HTML → Markdown sections) → remove template text → chunk → embed → data/index/
```

| Module | Responsibility |
|---|---|
| `placidbot/sources.py` | The seven allowed URLs and their page types |
| `placidbot/crawler.py` | Polite fetching; `LocalHtmlCrawler` reads saved HTML if the site blocks bots |
| `placidbot/extract.py` | HTML cleaning, main-content detection, Markdown, sections, schema.org prices |
| `placidbot/chunker.py` | Section-bounded chunks with a page/section header |
| `placidbot/index.py` | Hybrid BM25 + embedding search, save/load |
| `placidbot/ingest.py` | Ingestion pipeline and refresh report (also the CLI) |
| `placidbot/llm.py` | The only module that calls Gemini: retries, model fallbacks, JSON output, embeddings |
| `placidbot/prompts.py` | Router and answer prompts and their JSON schemas |
| `placidbot/guard.py` | Checks money amounts against the cited text |
| `placidbot/bot.py` | The pipeline above |
| `placidbot/ui.py`, `app.py` | Gradio interface and start-up |

## Design choices

- **Gemini** has a free tier, answers well in many languages, supports JSON-schema output and
  provides embeddings, so one provider covers everything. Model names are settings; switching
  provider means replacing `llm.py` only.
- **Two models.** A small one routes and rewrites the question; a stronger one writes the answer.
  When a model's free quota runs out, the next one in the fallback list is used.
- **Hybrid retrieval.** BM25 is exact on names and numbers ("adrenal", "Out-Patient", "$30,000");
  embeddings handle paraphrases and other languages. With seven pages a vector database is not
  needed: the index is a JSON file plus a NumPy array. Two small boosts help ranking: priced passages
  for cost questions, and section headings that match the question (every chunk of a package page
  repeats the page title, so the heading is what separates "Included in the In-Patient Program" from
  the FAQ).
- **Checks in code, not only in the prompt.** Amounts are compared as digits, so `$18,995`,
  `18995 USD` and Spanish `18.995 USD` match, while a total the model computed itself does not.
- **Citations by id.** The model can only cite passages it was shown and cannot invent a URL.
- **Refresh without downtime.** A new index is built next to the old one and swapped in only when
  the build succeeds.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                # add GEMINI_API_KEY and ADMIN_TOKEN
python -m placidbot.ingest                          # fetch the 7 pages and build data/index
python app.py                                       # http://127.0.0.1:7860
```

Tests and checks:

```bash
pytest                                     # 54 offline tests, no API key needed
python scripts/run_test_questions.py       # runs the test questions, writes docs/TEST_RESULTS.md
python scripts/eval_retrieval.py --bm25    # rank of the right passage for 16 questions
```

If the site blocks automated requests, save each page from a browser as
`html_snapshots/<source-id>.html` (ids are in `placidbot/sources.py`) and run
`python -m placidbot.ingest --html-dir html_snapshots`.

The extracted text is in `data/pages/*.md`; that is exactly what the bot knows.

## Deploy to Hugging Face Spaces

1. Create a Gradio Space. The demo runs on free **ZeroGPU** hardware, which requires Python 3.10 or
   3.12 (set in the header of this README) and at least one `@spaces.GPU` function. `app.py`
   declares an unused placeholder for this; the app itself needs no GPU. On CPU hardware the
   placeholder is skipped.
2. Under **Settings → Variables and secrets** add the **secrets** `GEMINI_API_KEY` and `ADMIN_TOKEN`.
3. Push this repository to the Space's `main` branch. `*.npy` files are stored with Git LFS.
4. `data/index/` and `data/pages/` are committed, so the app starts without crawling. Without them
   the first start builds the knowledge base (about a minute).

Free Spaces sleep after a period without visitors and take about a minute to wake up.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | required | Google AI Studio key |
| `ADMIN_TOKEN` | empty = admin disabled | Unlocks Refresh and the unanswered-questions view |
| `GEMINI_ANSWER_MODEL` | `gemini-3.6-flash` | Writes answers |
| `GEMINI_ANSWER_FALLBACKS` | `gemini-3.8-flash,…` | Tried in order when the answer model's quota runs out |
| `GEMINI_ROUTER_MODEL` | `gemini-3.5-flash-lite` | Rewrites follow-ups, detects language and intent |
| `GEMINI_ROUTER_FALLBACKS` | `gemini-3.1-flash-lite,…` | Same, for the router |
| `GEMINI_THINKING_LEVEL` | `LOW` | Thinking budget; `LOW` answers faster with the same results. Empty = model default |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Embeddings (768 dimensions) |
| `CRAWLER_CONTACT` | maintainer email | Shown in the crawler's User-Agent |
| `TOP_K` | `6` | Passages given to the answer model |

API keys are never in the code; they come from environment variables or Space secrets. Quote-form
leads and question logs are written to `data/logs/`, which is git-ignored.

## Limitations

- **Seven pages only.** Most real questions ("hip replacement in Spain") are outside this content,
  so the bot answers that it does not have the information and points to the free quote.
- **Only prices are checked in code.** Other facts (durations, inclusions) rely on the prompt and
  on citations.
- **Generic HTML extraction.** The extractor uses general rules rather than PlacidWay-specific
  selectors, so a layout change could let template text in. `data/pages/` shows the result.
- **Non-English questions** are searched through an English rewrite, so answer quality depends on
  that translation.
- **No persistent storage on the free Space.** Logs and leads are lost on restart, and an admin
  refresh lasts until the next restart.
- **Free-tier speed and limits.** Each message makes two or three Gemini calls and usually takes
  6–10 seconds. Under heavy use the free quota can run out; the bot then falls back to other models
  or asks the visitor to try again.

## Next steps

1. A larger evaluation set (50–100 questions with expected sources and "should refuse" labels) run
   in CI.
2. Claim-level verification: match every sentence of an answer to a supporting quote, not only prices.
3. Parsers for PlacidWay's package, price-list and comparison templates, so cost and comparison
   questions can be answered from structured records with exact filters.
4. Persistent storage for logs, leads and the index, plus a scheduled refresh that alerts on changes.
5. Send quote requests to PlacidWay's CRM or email, with spam protection.
6. Streaming replies and an embeddable widget for PlacidWay pages.
