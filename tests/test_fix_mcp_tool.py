"""Unit tests for the propose_fix MCP tool."""

from __future__ import annotations

import json

import pytest

from backend.agent.fix_generator import FixGenerator
from backend.mcp.tools import make_propose_fix_tool


class _StaticLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    @property
    def model(self) -> str:
        return "static-fake"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


VALID_DIFF = (
    "--- a/src/app.py\n"
    "+++ b/src/app.py\n"
    "@@ -1,1 +1,1 @@\n"
    "-return a - b\n"
    "+return a + b\n"
)


def _valid_response() -> str:
    return json.dumps({
        "explanation": "swap - for +",
        "diff": VALID_DIFF,
        "confidence": 0.8,
    })


def test_propose_fix_tool_returns_summary() -> None:
    gen = FixGenerator(llm=_StaticLLM(_valid_response()))
    tool = make_propose_fix_tool(gen)
    out = tool.execute({"goal": "fix add", "context": "some code"})
    assert "Goal: fix add" in out
    assert "Model: static-fake" in out
    assert "Valid diff: True" in out
    assert "Diff:" in out


def test_propose_fix_tool_requires_goal() -> None:
    gen = FixGenerator(llm=_StaticLLM(_valid_response()))
    tool = make_propose_fix_tool(gen)
    with pytest.raises(ValueError):
        tool.execute({})
    with pytest.raises(ValueError):
        tool.execute({"goal": "   ", "context": ""})


def test_propose_fix_tool_context_is_optional() -> None:
    gen = FixGenerator(llm=_StaticLLM(_valid_response()))
    tool = make_propose_fix_tool(gen)
    out = tool.execute({"goal": "fix add"})
    assert "Goal: fix add" in out