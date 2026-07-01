"""Unit tests for backend.agent.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.agent.models import Plan, PlanStep, StepKind, StepStatus


def test_step_defaults_to_pending() -> None:
    step = PlanStep(id=1, kind=StepKind.RETRIEVE, description="fetch")
    assert step.status == StepStatus.PENDING
    assert step.depends_on == ()


def test_step_rejects_zero_id() -> None:
    with pytest.raises(ValidationError):
        PlanStep(id=0, kind=StepKind.RETRIEVE, description="x")


def test_step_rejects_empty_description() -> None:
    with pytest.raises(ValidationError):
        PlanStep(id=1, kind=StepKind.ANALYZE, description="")


def test_step_is_frozen() -> None:
    step = PlanStep(id=1, kind=StepKind.RETRIEVE, description="x")
    with pytest.raises(ValidationError):
        step.description = "changed"  # type: ignore[misc]


def test_plan_step_count_matches() -> None:
    plan = Plan(
        goal="do stuff",
        steps=(
            PlanStep(id=1, kind=StepKind.RETRIEVE, description="a"),
            PlanStep(id=2, kind=StepKind.ANALYZE, description="b"),
        ),
    )
    assert plan.step_count == 2
    assert plan.kinds() == (StepKind.RETRIEVE, StepKind.ANALYZE)


def test_plan_rejects_empty_goal() -> None:
    with pytest.raises(ValidationError):
        Plan(goal="", steps=())


def test_plan_default_strategy_is_rule_based() -> None:
    plan = Plan(goal="x", steps=())
    assert plan.strategy == "rule_based"


def test_step_kind_values() -> None:
    values = {k.value for k in StepKind}
    assert values == {
        "retrieve", "analyze", "github_read", "generate",
        "validate", "human_approval", "draft_pr",
    }


def test_step_status_values() -> None:
    values = {s.value for s in StepStatus}
    assert values == {"pending", "in_progress", "completed", "failed", "skipped"}