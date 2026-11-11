"""Entry point (Hugging Face Spaces runs this file).

On start-up the app loads the saved knowledge base from data/index/. If none
exists yet (first deploy), it crawls the seven PlacidWay pages and builds one.
"""

from __future__ import annotations

import logging

from placidbot.config import get_settings
from placidbot.index import HybridIndex
from placidbot.ingest import build_index
from placidbot.llm import GeminiClient, LLMError
from placidbot.ui import App

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

settings = get_settings()

try:
    llm = GeminiClient.from_settings(settings)
except LLMError as exc:
    log.error("LLM not configured (%s). Set the GEMINI_API_KEY secret.", exc)
    llm = None

index = HybridIndex.load(settings.index_dir)
if index is None:
    log.info("No saved knowledge base found; building it now (takes about a minute).")
    try:
        index, report = build_index(settings, llm)
        log.info(report.summary())
    except Exception as exc:  # the UI still starts and explains the problem
        log.error("Initial ingestion failed: %s", exc)
        index = None

try:  # the Space runs on ZeroGPU (free accounts cannot pick CPU basic), which refuses to start without a GPU function
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_placeholder() -> None:
        """Never called: all model work happens in the Gemini API."""
except ImportError:  # local runs and CPU Spaces
    pass

demo = App(settings, llm, index).build()

if __name__ == "__main__":
    demo.launch()
