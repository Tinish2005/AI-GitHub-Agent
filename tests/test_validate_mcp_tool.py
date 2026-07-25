"""Unit tests for the validate_fix MCP tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agent.fix_generator import FixGenerator
from backend.agent.validation_checks import SyntaxCheck
from backend.agent.validation_pipeline import ValidationPipeline
from backend.mcp.tools import make_validate_fix_tool


class _StaticLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    @property
    def model(self) -> str:
        return "static-fake"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


VALID_DIFF = (
    "--- /dev/null\n"
    "+++ b/hello.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+def greet():\n"
    "+    return 'hi'\n"
)


def _valid_response() -> str:
    return json.dumps({"explanation": "add greet", "diff": VALID_DIFF, "confidence": 0.9})


def test_validate_fix_tool_returns_summary(tmp_path: Path) -> None:
    gen = FixGenerator(llm=_StaticLLM(_valid_response()))
    validator = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    tool = make_validate_fix_tool(gen, validator)
    out = tool.execute({"goal": "add greet", "context": ""})
    assert "Validation for goal: add greet" in out
    assert "Proposal valid: True" in out
    assert "Validation passed: True" in out
    assert "[pass] syntax" in out


def test_validate_fix_tool_requires_goal(tmp_path: Path) -> None:
    gen = FixGenerator(llm=_StaticLLM(_valid_response()))
    validator = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    tool = make_validate_fix_tool(gen, validator)
    with pytest.raises(ValueError):
        tool.execute({})
    with pytest.raises(ValueError):
        tool.execute({"goal": "   "})