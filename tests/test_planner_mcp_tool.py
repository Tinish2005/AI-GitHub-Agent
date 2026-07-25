"""Unit tests for the create_plan MCP tool."""

from __future__ import annotations

import pytest

from backend.agent.planner import RuleBasedPlanner
from backend.mcp.tools import make_create_plan_tool


def test_create_plan_tool_returns_summary() -> None:
    tool = make_create_plan_tool(RuleBasedPlanner())
    out = tool.execute({"goal": "Fix the bug in login"})
    assert "Plan for goal:" in out
    assert "Steps" in out
    assert "[retrieve]" in out
    assert "[draft_pr]" in out


def test_create_plan_tool_requires_goal() -> None:
    tool = make_create_plan_tool(RuleBasedPlanner())
    with pytest.raises(ValueError):
        tool.execute({})
    with pytest.raises(ValueError):
        tool.execute({"goal": "   "})


def test_create_plan_tool_default_strategy_is_rule_based() -> None:
    tool = make_create_plan_tool(RuleBasedPlanner())
    out = tool.execute({"goal": "review auth"})
    assert "Strategy: rule_based" in out