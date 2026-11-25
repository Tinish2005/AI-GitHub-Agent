"""Offline test of scripts/live_check_llm.py (a stub AI, no network)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "live_check_llm.py"


def _load():
    spec = importlib.util.spec_from_file_location("live_check_llm", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stub:
    model = "stub"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.reply


GOOD_DIFF = "--- a/calc.py\n+++ b/calc.py\n@@ -1,3 +1,3 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n \n"


def levels(results: list) -> list:
    return [level for level, _ in results]


def test_a_good_answer_passes_every_check() -> None:
    reply = json.dumps({"explanation": "fix", "diff": GOOD_DIFF, "confidence": 0.9})
    results = _load().run_check(Stub(reply))
    assert "FAIL" not in levels(results) and levels(results).count("PASS") == 3


def test_an_answer_that_is_not_json_fails() -> None:
    assert "FAIL" in levels(_load().run_check(Stub("Sure! Here is the fix.")))


def test_a_diff_for_code_that_is_not_there_fails() -> None:
    wrong = GOOD_DIFF.replace("return a - b\n+", "return x * y\n+", 1)
    results = _load().run_check(Stub(json.dumps({"explanation": "e", "diff": wrong})))
    assert levels(results).count("FAIL") >= 1 and any("does not fit" in m for _, m in results)


def test_a_diff_that_does_not_make_the_change_fails() -> None:
    other = GOOD_DIFF.replace("a + b", "a * b")
    results = _load().run_check(Stub(json.dumps({"explanation": "e", "diff": other})))
    assert any(level == "FAIL" and "a + b" in m for level, m in results)
