---
title: PlacidWay Knowledge Chatbot
emoji: 💬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# PlacidWay Website Knowledge Chatbot

A chatbot that answers visitor questions using **only** seven PlacidWay pages, cites the page every
answer comes from, and says "I don't know" instead of guessing.

- **Live demo:** `<your Hugging Face Space URL>`
- **Test results:** [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md)
- **Limitations and next steps:** [docs/WRITEUP.md](docs/WRITEUP.md)

## What it does

| Requirement | How |
|---|---|
| Content ingestion | `placidbot/crawler.py` fetches the 7 assigned URLs only (no link following), checks robots.txt, sends a descriptive User-Agent, waits 1 s between requests. |
| Knowledge base | Pages are cleaned to Markdown, split by heading into chunks (tables kept whole, long tables split with their header repeated), and indexed with hybrid BM25 + embedding search. |
| Grounded answers | The model sees only the retrieved passages and must return the ids of those it used. The app turns the ids into source links; the model never writes URLs. |
| Honest "I don't know" | Prompt rule plus code checks. An answer with facts but no valid citation, or with a price that is not in the cited text, is sent back for one rewrite and otherwise replaced by a fixed "I don't have that information, request a free quote" reply. |
| Cost questions | Prices are copied with currency and page context ("starting from", program, duration, inclusions). The model may not calculate, convert or estimate. |
| Chat UI | Gradio, with examples, a free-quote form and a knowledge-base panel. |
| Refresh | `python -m placidbot.ingest` or the admin "Refresh content" button. The report lists pages that changed; a page that fails to fetch keeps its previous content. |
| Live deployment | Hugging Face Spaces (free CPU), see below. |

Nice-to-haves included: replies in the user's language, follow-up questions using chat history, a
medical disclaimer shown once per conversation, lead capture (quote form), and a log of unanswered
questions viewable in the admin panel.

## Architecture

```
            ┌────────────── ingestion (on start-up if no index, CLI, or admin Refresh) ─────────────┐
 7 URLs ──▶ PoliteCrawler ──▶ extract_page ──▶ drop_cross_page_boilerplate ──▶ chunk_page ──▶ embed ──▶ data/index/
            robots.txt,       HTML → Markdown   template lines found          heading-aware      Gemini     chunks.json
            1 req/s, UA       sections by       on ≥4 pages                   chunks with        embedding  embeddings.npy
                              heading           (numbers kept)                page/section header           meta.json
            └──────────────────────────────────────────────────────────────────────────────────────────────┘

 user message + history
   │
   ▼
 1. Router (gemini flash-lite, JSON) ── standalone English question · language · intent
   │                                    (question / quote_request / booking / medical_advice / out_of_scope / greeting)
   ▼
 2. Retrieve: hybrid BM25 + embeddings, top 6 (skipped for booking / out-of-scope / greeting)
   ▼
 3. Answer (gemini flash, JSON) ── {status, answer, source_ids, medical_topic}
   ▼
 4. Verify in code ── cited ids exist? every money amount present in the cited passages?
   │                   fail → one rewrite with the problem stated → still failing → safe fallback
   ▼
 5. Compose ── answer + source links + disclaimer (first medical reply only)
   ▼
 6. Log ── not_found / partial / guard-triggered questions → data/logs/unanswered.jsonl
```

| Module | Responsibility |
|---|---|
| `placidbot/sources.py` | The seven allowed URLs and their page types |
| `placidbot/crawler.py` | Polite fetching; `LocalHtmlCrawler` reads saved HTML if the site blocks bots |
| `placidbot/extract.py` | HTML cleaning, main-content detection, Markdown, sections, schema.org prices |
| `placidbot/chunker.py` | Section-bounded chunks with a page/section header |
| `placidbot/index.py` | Hybrid BM25 + embedding search, save/load |
| `placidbot/ingest.py` | Ingestion pipeline and refresh report (also the CLI) |
| `placidbot/llm.py` | The only module that calls Gemini (retries, JSON output, embeddings) |
| `placidbot/prompts.py` | Router and answer prompts and their JSON schemas |
| `placidbot/guard.py` | Money-amount verification against cited text |
| `placidbot/bot.py` | The chat pipeline above |
| `placidbot/ui.py`, `app.py` | Gradio interface and start-up |

### Design choices

- **Gemini.** Free tier for the assessment, strong multilingual answers, native JSON-schema output
  (no fragile parsing), and one provider for both chat and embeddings. Model names are settings, so
  switching model is a config change; switching provider means replacing `llm.py` only.
- **A small model for routing, a stronger one for answering.** Routing is classification and
  rewriting; the answer is where accuracy matters.
- **Hybrid retrieval.** BM25 is exact on names and numbers ("adrenal", "Out-Patient", "$30,000");
  embeddings catch paraphrases and non-English questions. With only seven pages a vector database
  would be overhead, so the index is a JSON file plus a NumPy array.
- **Code checks, not only prompt rules.** The rubric weights accuracy and honesty highest, so the
  promise "never invents prices" is enforced after generation: amounts are compared as digits,
  so `$18,995`, `18995 USD` and Spanish `18.995 USD` match, while a computed total does not.
- **Citations by id.** The model chooses passage ids; the code maps them to URLs. A model cannot
  cite a page it was not shown, and it cannot invent a URL.
- **Refresh without downtime.** A new index is built beside the old one and swapped in only when it
  succeeds.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                  # put GEMINI_API_KEY and ADMIN_TOKEN in .env
python -m placidbot.ingest                            # fetch the 7 pages, build data/index
python app.py                                         # open http://127.0.0.1:7860
pytest                                                # 35 offline tests, no API key needed
python scripts/run_test_questions.py                  # writes docs/TEST_RESULTS.md
```

If the site refuses automated requests, open each page in a browser, save it as
`html_snapshots/<source-id>.html` (ids are in `placidbot/sources.py`) and run
`python -m placidbot.ingest --html-dir html_snapshots`.

Check what was extracted in `data/pages/*.md`: that is exactly what the bot knows.

## Deploy to Hugging Face Spaces

1. Create a Space: SDK **Gradio**, hardware **CPU basic (free)**, visibility **public**.
2. In the Space's **Settings → Variables and secrets** add the secrets `GEMINI_API_KEY` and `ADMIN_TOKEN`.
3. Push this repository to the Space (`git remote add space https://huggingface.co/spaces/<user>/<space>`,
   then `git push space main`). This README's header tells Spaces how to run `app.py`.
4. On first start the app builds the knowledge base (about a minute). Committing `data/index/` and
   `data/pages/` after a local ingest makes restarts instant and lets reviewers read the extracted text.

Free Spaces sleep after a long period without visitors and wake on the next visit (about a minute).

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Google AI Studio key |
| `ADMIN_TOKEN` | (empty = admin disabled) | Unlocks Refresh and the unanswered-questions view |
| `GEMINI_ANSWER_MODEL` | `gemini-3.8-flash` | Writes answers |
| `GEMINI_ROUTER_MODEL` | `gemini-3.5-flash-lite` | Rewrites follow-ups, detects language and intent |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Embeddings (768 dimensions) |
| `CRAWLER_CONTACT` | maintainer email | Shown in the crawler's User-Agent |
| `TOP_K` | `6` | Passages given to the answer model |

## Privacy and secrets

No keys are in the code; they come from environment variables or Space secrets. Quote-form leads
and question logs are written to `data/logs/`, which is git-ignored.
