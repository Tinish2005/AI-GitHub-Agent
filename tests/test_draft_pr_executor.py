"""Unit tests for DraftPRExecutor."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.agent.draft_pr_service import FakeDraftPRService
from backend.agent.executor import DraftPRExecutor, StepContext
from backend.agent.fix_models import FixHunk, FixProposal
from backend.agent.models import PlanStep, StepKind
from backend.agent.validation_checks import SyntaxCheck
from backend.agent.validation_models import CheckResult, ValidationResult


VALID_DIFF = (
    "--- /dev/null\n"
    "+++ b/hello.py\n"
    "@@ -0,0 +1,1 @@\n"
    "+def greet():\n"
)


def _proposal() -> FixProposal:
    return FixProposal(
        goal="add greet",
        explanation="new file",
        diff=VALID_DIFF,
        hunks=(FixHunk(file_path="hello.py", added_lines=1, removed_lines=0),),
        model="fake",
        confidence=0.9,
        is_valid=True,
    )


def _validation(passed: bool = True, score: float = 1.0) -> ValidationResult:
    return ValidationResult(
        proposal_goal="add greet",
        checks=(CheckResult(name="syntax", passed=True),),
        passed=passed,
        score=score,
    )


def _step() -> PlanStep:
    return PlanStep(id=7, kind=StepKind.DRAFT_PR, description="open the PR")


def test_executor_creates_pr_when_all_conditions_met() -> None:
    ex = DraftPRExecutor(
        FakeDraftPRService(next_number=101),
        owner="Tinish2005", repo="AI-GitHub-Agent",
    )
    ctx = StepContext(
        goal="add greet",
        step=_step(),
        prior_outputs={"fix_proposal": _proposal(), "validation_result": _validation()},
    )
    out = ex.run(ctx)
    assert "Draft PR (created=True)" in out
    assert "#101" in out
    assert "agent/add-greet" in out


def test_executor_skips_when_validation_failed() -> None:
    ex = DraftPRExecutor(
        FakeDraftPRService(),
        owner="Tinish2005", repo="AI-GitHub-Agent",
    )
    ctx = StepContext(
        goal="add greet",
        step=_step(),
        prior_outputs={
            "fix_proposal": _proposal(),
            "validation_result": _validation(passed=False, score=0.2),
        },
    )
    out = ex.run(ctx)
    assert "Draft PR (created=False)" in out
    assert "skipped" in out


def test_executor_needs_proposal() -> None:
    ex = DraftPRExecutor(
        FakeDraftPRService(),
        owner="Tinish2005", repo="AI-GitHub-Agent",
    )
    ctx = StepContext(goal="x", step=_step(), prior_outputs={})
    out = ex.run(ctx)
    assert "no FixProposal" in out


def test_executor_needs_validation() -> None:
    ex = DraftPRExecutor(
        FakeDraftPRService(),
        owner="Tinish2005", repo="AI-GitHub-Agent",
    )
    ctx = StepContext(
        goal="x", step=_step(),
        prior_outputs={"fix_proposal": _proposal()},
    )
    out = ex.run(ctx)
    assert "no ValidationResult" in out


def test_executor_rejects_missing_owner() -> None:
    with pytest.raises(ValueError):
        DraftPRExecutor(FakeDraftPRService(), owner="", repo="r")