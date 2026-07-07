"""Unit tests for the draft_pr MCP tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agent.draft_pr_service import FakeDraftPRService
from backend.agent.fix_generator import FixGenerator
from backend.agent.validation_checks import SyntaxCheck
from backend.agent.validation_pipeline import ValidationPipeline
from backend.mcp.tools import make_draft_pr_tool


class _StaticLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    @property
    def model(self) -> str:
        return "static-fake"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


# Valid two-line diff. Header @@ -0,0 +1,2 @@ matches 2 added lines.
VALID_DIFF = (
    "--- /dev/null\n"
    "+++ b/hello.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+def greet():\n"
    "+    return 'hi'\n"
)


def _valid_response() -> str:
    return json.dumps({"explanation": "add greet", "diff": VALID_DIFF, "confidence": 0.9})


def test_draft_pr_tool_creates_pr(tmp_path: Path) -> None:
    gen = FixGenerator(llm=_StaticLLM(_valid_response()))
    validator = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    svc = FakeDraftPRService(next_number=200)
    tool = make_draft_pr_tool(gen, validator, svc)
    out = tool.execute({
        "goal": "add greet", "context": "",
        "owner": "Tinish2005", "repo": "AI-GitHub-Agent",
    })
    assert "created: True" in out
    assert "#200" in out


def test_draft_pr_tool_requires_goal(tmp_path: Path) -> None:
    gen = FixGenerator(llm=_StaticLLM(_valid_response()))
    validator = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    svc = FakeDraftPRService()
    tool = make_draft_pr_tool(gen, validator, svc)
    with pytest.raises(ValueError):
        tool.execute({"owner": "o", "repo": "r"})


def test_draft_pr_tool_requires_owner_repo(tmp_path: Path) -> None:
    gen = FixGenerator(llm=_StaticLLM(_valid_response()))
    validator = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    svc = FakeDraftPRService()
    tool = make_draft_pr_tool(gen, validator, svc)
    with pytest.raises(ValueError):
        tool.execute({"goal": "x"})
    with pytest.raises(ValueError):
        tool.execute({"goal": "x", "owner": "o"})