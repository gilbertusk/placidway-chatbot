"""Append-only JSONL logs: unanswered questions (content gaps) and quote leads."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()


class JsonlLog:
    def __init__(self, path: Path):
        self.path = path

    def append(self, record: dict) -> None:
        record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        with _lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read(self, limit: int = 200) -> list[dict]:
        if not self.path.exists():
            return []
        with _lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        records = []
        for line in lines[-limit:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records
