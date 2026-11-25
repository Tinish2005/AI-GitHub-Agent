"""Offline test of the live-check script's scoring logic (no Gemini calls)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "live_check_execute_context.py"


def _load():
    spec = importlib.util.spec_from_file_location("live_check_execute_context", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Store:
    """Returns the expected file and function for every case, or junk."""

    def __init__(self, good: bool) -> None:
        self.good = good
        self.by_question: dict = {}

    def query(self, text: str, *, top_k: int = 5) -> list:
        if self.good:
            case = next(c for c in _load().CASES if c["question"] == text)
            meta = {"qualified_name": case["names"][0], "file_path": "src\\requests\\" + case["files"][0]}
        else:
            meta = {"qualified_name": "FlaskyStyle", "file_path": "docs/_themes/x.py"}
        return [SimpleNamespace(metadata=meta, document="code", distance=0.2)]


def test_script_passes_when_right_files_and_functions_come_back() -> None:
    report = _load().evaluate(_Store(good=True))
    assert report["passed"] is True
    assert (report["file_hits"], report["function_hits"]) == (5, 5)


def test_script_fails_and_marks_irrelevant_results_when_search_is_wrong() -> None:
    report = _load().evaluate(_Store(good=False))
    assert report["passed"] is False
    assert report["file_hits"] == 0
    assert all(not h["relevant"] for row in report["rows"] for h in row["hits"])
