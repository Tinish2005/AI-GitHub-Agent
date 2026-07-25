"""Unit tests for LLMPlanner (using fake LLMs)."""

from __future__ import annotations

import json

import pytest

from backend.agent.models import StepKind
from backend.agent.planner import LLMPlanner


class _StaticLLM:
    """Fake LLM that always returns the same string."""

    def __init__(self, response: str, model: str = "static-fake") -> None:
        self._response = response
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


def _valid_json_plan() -> str:
    return json.dumps({
        "steps": [
            {"kind": "retrieve", "description": "find code"},
            {"kind": "analyze", "description": "reason over it"},
            {"kind": "generate", "description": "produce answer"},
        ]
    })


def test_llm_planner_parses_valid_json() -> None:
    planner = LLMPlanner(llm=_StaticLLM(_valid_json_plan()))
    plan = planner.plan("do the thing")
    assert plan.strategy == "llm"
    assert plan.step_count == 3
    assert plan.kinds() == (StepKind.RETRIEVE, StepKind.ANALYZE, StepKind.GENERATE)


def test_llm_planner_handles_json_wrapped_in_prose() -> None:
    wrapped = "Sure! Here is the plan:\n\n" + _valid_json_plan() + "\n\nHope this helps."
    planner = LLMPlanner(llm=_StaticLLM(wrapped))
    plan = planner.plan("goal")
    assert plan.step_count == 3


def test_llm_planner_falls_back_on_bad_json() -> None:
    planner = LLMPlanner(llm=_StaticLLM("this is not json at all"))
    plan = planner.plan("Fix the login bug")
    # Fallback plan should be from RuleBasedPlanner - fix-bug template.
    assert plan.strategy == "rule_based"
    assert plan.steps[-1].kind == StepKind.DRAFT_PR


def test_llm_planner_falls_back_on_missing_steps() -> None:
    bad = json.dumps({"not_steps": []})
    planner = LLMPlanner(llm=_StaticLLM(bad))
    plan = planner.plan("review the auth module")
    assert plan.strategy == "rule_based"


def test_llm_planner_falls_back_on_unknown_step_kind() -> None:
    bad = json.dumps({
        "steps": [{"kind": "teleport", "description": "beam me up"}]
    })
    planner = LLMPlanner(llm=_StaticLLM(bad))
    plan = planner.plan("review the auth module")
    assert plan.strategy == "rule_based"


def test_llm_planner_rejects_empty_goal() -> None:
    planner = LLMPlanner(llm=_StaticLLM(_valid_json_plan()))
    with pytest.raises(ValueError):
        planner.plan("")


def test_llm_planner_strategy_label_when_success() -> None:
    planner = LLMPlanner(llm=_StaticLLM(_valid_json_plan()))
    plan = planner.plan("anything")
    assert plan.strategy == "llm"