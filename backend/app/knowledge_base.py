"""Loads and exposes the structured biodiversity/soil/climate knowledge layer.

The knowledge layer is intentionally a structured JSON dataset (rather than
raw prose dumped into a prompt) so that every entry has machine-readable
trigger conditions, impacted metrics, a source citation, and a confidence
rating. This is what the retriever indexes and what the reasoning engine
filters against multi-metric input.
"""
import json
from pathlib import Path
from typing import List, Dict, Any

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge.json"


class KnowledgeBase:
    def __init__(self, path: Path = DATA_PATH):
        self.path = path
        self.entries: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def all(self) -> List[Dict[str, Any]]:
        return self.entries

    def searchable_text(self, entry: Dict[str, Any]) -> str:
        """Flatten an entry into the text blob the retriever indexes."""
        return " ".join([
            entry.get("title", ""),
            entry.get("category", ""),
            entry.get("intervention", ""),
            entry.get("mechanism", ""),
            " ".join(entry.get("impacted_metrics", [])),
            " ".join(entry.get("trigger_metrics", [])),
        ])

    def by_id(self, entry_id: str) -> Dict[str, Any]:
        for e in self.entries:
            if e["id"] == entry_id:
                return e
        raise KeyError(entry_id)


knowledge_base = KnowledgeBase()
