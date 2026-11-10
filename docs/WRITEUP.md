# Write-up: limitations and next steps

## Known limitations

- **Seven pages only.** Most real questions ("hip replacement in Spain", "dental implants in Mexico")
  are outside this content, so the bot correctly answers "I don't know" and points to the free quote.
  It is a demonstration of grounding, not a full PlacidWay assistant.
- **Price check covers money, not every fact.** Every amount in an answer must appear in a cited
  passage, but other facts (durations, inclusions) rely on the prompt and on citations. A wrong
  non-numeric statement would not be caught by code.
- **Generic HTML extraction.** The extractor uses general rules (remove navigation/footers/forms,
  find the main block, drop lines repeated on 4+ pages) rather than PlacidWay-specific selectors.
  A layout change could let some template text in or, less likely, drop content. The extracted text
  is saved in `data/pages/` so it can be checked by eye.
- **Language.** Retrieval runs on an English rewrite of the question; the quality of non-English
  answers depends on the router's translation.
- **Ephemeral storage.** On free Hugging Face Spaces, logs and leads are lost when the Space restarts,
  and a refresh made from the admin panel lasts until the next restart (the committed index is used).
- **Rate limits.** Each message makes two or three Gemini calls; heavy use on the free tier can hit
  limits. The app then replies with a friendly "try again" message instead of failing.

## With another week

1. **An evaluation set.** 50 to 100 questions with expected sources and "should refuse" labels, run
   in CI, so prompt or chunking changes are measured instead of eyeballed.
2. **Claim-level verification.** Extend the code check from prices to every sentence: ask a small
   model to match each claim to a supporting quote and drop unsupported claims (as in my earlier
   Telaah project, which reached 98.5% grounding this way).
3. **Site-specific parsers** for PlacidWay's package, price-list and comparison templates, producing
   structured records (treatment, clinic, city, country, price, currency, qualifier). Cost and
   comparison questions could then be answered from a table with exact filters, not only text search.
4. **Persistent storage** for logs, leads and the index (a small database or object storage), plus a
   scheduled refresh that alerts when a page changes or fails.
5. **Lead hand-off.** Send quote requests to PlacidWay's CRM or email instead of a local file, with
   spam protection.
6. **Streaming replies** and a lighter UI that could be embedded on PlacidWay pages as a widget.
