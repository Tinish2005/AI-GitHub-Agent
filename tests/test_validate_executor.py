"""Unit tests for ValidateExecutor."""

from __future__ import annotations

from pathlib import Path

from backend.agent.executor import StepContext, ValidateExecutor
from backend.agent.fix_models import FixHunk, FixProposal
from backend.agent.models import PlanStep, StepKind
from backend.agent.validation_checks import SyntaxCheck
from backend.agent.validation_pipeline import ValidationPipeline


VALID_DIFF = (
    "--- /dev/null\n"
    "+++ b/hello.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+def greet():\n"
    "+    return 'hi'\n"
)


def _proposal() -> FixProposal:
    return FixProposal(
        goal="add greet",
        explanation="new file",
        diff=VALID_DIFF,
        hunks=(FixHunk(file_path="hello.py", added_lines=2, removed_lines=0),),
        model="fake",
        confidence=0.8,
        is_valid=True,
    )


def _step() -> PlanStep:
    return PlanStep(id=5, kind=StepKind.VALIDATE, description="validate")


def test_validate_executor_runs_pipeline(tmp_path: Path) -> None:
    pipeline = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    ex = ValidateExecutor(pipeline)
    ctx = StepContext(
        goal="add greet",
        step=_step(),
        prior_outputs={"fix_proposal": _proposal()},
    )
    out = ex.run(ctx)
    assert "Validation" in out
    assert "passed=True" in out
    assert "[pass] syntax" in out


def test_validate_executor_missing_proposal(tmp_path: Path) -> None:
    pipeline = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    ex = ValidateExecutor(pipeline)
    ctx = StepContext(goal="x", step=_step(), prior_outputs={})
    out = ex.run(ctx)
    assert "no FixProposal" in out