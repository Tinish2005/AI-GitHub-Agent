"""A diff that ends with a blank context line must not be cut short (found in Loop 8)."""

from __future__ import annotations

import json

from backend.agent.fix_generator import FixGenerator

DIFF_ENDING_IN_A_BLANK_CONTEXT_LINE = (
    "--- a/calc.py\n+++ b/calc.py\n@@ -1,3 +1,3 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n \n"
)


class Stub:
    model = "stub"

    def __init__(self, diff: str) -> None:
        self.reply = json.dumps({"explanation": "fix", "diff": diff, "confidence": 0.9})

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.reply


def test_a_trailing_blank_context_line_is_kept_and_the_diff_is_valid() -> None:
    proposal = FixGenerator(llm=Stub(DIFF_ENDING_IN_A_BLANK_CONTEXT_LINE)).propose("goal", "ctx")
    assert proposal.is_valid, proposal.validation_error
    assert proposal.diff.endswith("\n ")          # the blank context line (one space) survived


def test_blank_lines_around_the_diff_are_still_removed() -> None:
    proposal = FixGenerator(llm=Stub("\n\n" + DIFF_ENDING_IN_A_BLANK_CONTEXT_LINE + "\n\n")).propose("goal", "ctx")
    assert proposal.diff.startswith("--- a/calc.py") and proposal.is_valid


def test_an_ordinary_diff_is_unchanged() -> None:
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b"
    assert FixGenerator(llm=Stub(diff + "\n")).propose("goal", "ctx").diff == diff
