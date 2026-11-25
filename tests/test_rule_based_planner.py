"""Unit tests for RuleBasedPlanner."""

from __future__ import annotations

import pytest

from backend.agent.models import Plan, StepKind
from backend.agent.planner import RuleBasedPlanner


def test_planner_rejects_empty_goal() -> None:
    p = RuleBasedPlanner()
    with pytest.raises(ValueError):
        p.plan("")
    with pytest.raises(ValueError):
        p.plan("   ")


def test_planner_strategy_label() -> None:
    p = RuleBasedPlanner()
    plan = p.plan("something")
    assert plan.strategy == "rule_based"


def test_fix_bug_plan_ends_with_validate() -> None:
    # Loop 5: approval and the draft PR are no longer plan steps (Execute cannot run them).
    p = RuleBasedPlanner()
    plan = p.plan("Fix the login bug that crashes on empty password")
    assert plan.steps[0].kind == StepKind.RETRIEVE
    assert plan.steps[-1].kind == StepKind.VALIDATE
    assert plan.kinds() == (
        StepKind.RETRIEVE, StepKind.ANALYZE, StepKind.GENERATE, StepKind.VALIDATE,
    )
    assert StepKind.HUMAN_APPROVAL not in plan.kinds()
    assert StepKind.DRAFT_PR not in plan.kinds()


def test_review_plan_has_retrieve_and_analyze() -> None:
    p = RuleBasedPlanner()
    plan = p.plan("Please review the authentication module")
    assert plan.kinds() == (StepKind.RETRIEVE, StepKind.ANALYZE)


def test_github_plan_reads_github_first() -> None:
    p = RuleBasedPlanner()
    plan = p.plan("List all open issues on the repo")
    assert plan.steps[0].kind == StepKind.GITHUB_READ


def test_pr_plan_ends_with_validate() -> None:
    p = RuleBasedPlanner()
    plan = p.plan("Propose a small PR that adds type hints")
    assert plan.steps[-1].kind == StepKind.VALIDATE
    assert StepKind.DRAFT_PR not in plan.kinds()


def test_default_plan_is_qa_shape() -> None:
    p = RuleBasedPlanner()
    plan = p.plan("What does the config module do?")
    assert plan.kinds() == (StepKind.RETRIEVE, StepKind.ANALYZE)


def test_step_ids_are_1_indexed_and_dense() -> None:
    p = RuleBasedPlanner()
    plan = p.plan("fix bug in payment flow")
    ids = [s.id for s in plan.steps]
    assert ids == list(range(1, len(ids) + 1))


def test_returns_plan_instance() -> None:
    p = RuleBasedPlanner()
    plan = p.plan("hello")
    assert isinstance(plan, Plan)
    assert plan.goal == "hello"