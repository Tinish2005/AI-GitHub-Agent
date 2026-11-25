"""Loop 1 tests: Execute passes data between steps and reports honestly."""

from __future__ import annotations

from pathlib import Path

from backend.agent.draft_pr_service import FakeDraftPRService
from backend.agent.engine import ExecutionEngine
from backend.agent.executor import (
    PATCH_MARKER,
    DraftPRExecutor,
    GenerateExecutor,
    PlannedExecutor,
    StepContext,
    StepSkipped,
    ValidateExecutor,
)
from backend.agent.fix_models import FixHunk, FixProposal
from backend.agent.models import Plan, PlanStep, StepKind, StepStatus
from backend.agent.validation_checks import SyntaxCheck
from backend.agent.validation_pipeline import ValidationPipeline
from backend.mcp.tools import make_execute_plan_tool
from backend.agent.planner import RuleBasedPlanner


DIFF = (
    "--- /dev/null\n"
    "+++ b/hello.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+def greet():\n"
    "+    return 'hi'\n"
)


class _StubGenerator:
    """Stands in for the LLM-backed FixGenerator."""

    def propose(self, goal: str, context: str) -> FixProposal:
        return FixProposal(
            goal=goal,
            explanation="adds a greet function",
            diff=DIFF,
            hunks=(FixHunk(file_path="hello.py", added_lines=2, removed_lines=0),),
            model="stub",
            confidence=0.9,
            is_valid=True,
        )


class _Stub:
    def __init__(self, kind: StepKind, output: str = "ok") -> None:
        self.kind = kind
        self.output = output

    def run(self, context: StepContext) -> str:
        return self.output


class _AlwaysSkips:
    def __init__(self, kind: StepKind) -> None:
        self.kind = kind
        self.calls = 0

    def run(self, context: StepContext) -> str:
        self.calls += 1
        raise StepSkipped("nothing to do")


def _plan(*kinds: StepKind) -> Plan:
    steps = tuple(PlanStep(id=i + 1, kind=k, description=f"step {i + 1}") for i, k in enumerate(kinds))
    return Plan(goal="fix the login bug", steps=steps, strategy="rule_based")


def _engine(tmp_path: Path, **kwargs) -> ExecutionEngine:
    pipeline = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    executors = {
        StepKind.RETRIEVE: _Stub(StepKind.RETRIEVE, "Retrieved 1 chunks"),
        StepKind.GENERATE: GenerateExecutor(_StubGenerator()),
        StepKind.VALIDATE: ValidateExecutor(pipeline),
        StepKind.HUMAN_APPROVAL: PlannedExecutor(StepKind.HUMAN_APPROVAL, "handled outside the engine"),
        StepKind.DRAFT_PR: PlannedExecutor(StepKind.DRAFT_PR, "not configured"),
    }
    return ExecutionEngine(executors, **kwargs)


FIX_PLAN = (
    StepKind.RETRIEVE, StepKind.GENERATE, StepKind.VALIDATE,
    StepKind.HUMAN_APPROVAL, StepKind.DRAFT_PR,
)


def test_generate_step_output_contains_the_patch(tmp_path: Path) -> None:
    result = _engine(tmp_path).run(_plan(*FIX_PLAN))
    out = result.steps[1].output
    assert PATCH_MARKER in out
    assert "+def greet():" in out
    assert "adds a greet function" in out


def test_validate_receives_the_generated_fix(tmp_path: Path) -> None:
    result = _engine(tmp_path).run(_plan(*FIX_PLAN))
    validate = result.steps[2]
    assert validate.status == StepStatus.COMPLETED
    assert "passed=True" in validate.output
    assert "[pass] syntax" in validate.output
    assert "skipping" not in validate.output


def test_placeholder_steps_are_skipped_with_a_reason(tmp_path: Path) -> None:
    result = _engine(tmp_path).run(_plan(*FIX_PLAN))
    human, pr = result.steps[3], result.steps[4]
    assert human.status == StepStatus.SKIPPED
    assert pr.status == StepStatus.SKIPPED
    assert human.error.startswith("[planned]")
    assert "not configured" in pr.error


def test_counts_add_up_to_total_steps(tmp_path: Path) -> None:
    result = _engine(tmp_path).run(_plan(*FIX_PLAN))
    assert (result.completed, result.skipped, result.failed) == (3, 2, 0)
    assert result.completed + result.skipped + result.failed == result.total_steps
    assert result.aborted is False


def test_validate_is_skipped_when_there_is_no_fix(tmp_path: Path) -> None:
    result = _engine(tmp_path).run(_plan(StepKind.VALIDATE))
    step = result.steps[0]
    assert step.status == StepStatus.SKIPPED
    assert "nothing to validate" in step.error
    assert (result.completed, result.skipped) == (0, 1)


def test_skipped_step_does_not_abort_later_steps() -> None:
    executors = {
        StepKind.HUMAN_APPROVAL: _AlwaysSkips(StepKind.HUMAN_APPROVAL),
        StepKind.RETRIEVE: _Stub(StepKind.RETRIEVE, "found"),
    }
    result = ExecutionEngine(executors, abort_on_failure=True).run(
        _plan(StepKind.HUMAN_APPROVAL, StepKind.RETRIEVE)
    )
    assert result.aborted is False
    assert result.steps[1].status == StepStatus.COMPLETED


def test_skipped_step_is_not_retried() -> None:
    skipper = _AlwaysSkips(StepKind.HUMAN_APPROVAL)
    result = ExecutionEngine({StepKind.HUMAN_APPROVAL: skipper}, max_retries=3).run(
        _plan(StepKind.HUMAN_APPROVAL)
    )
    assert skipper.calls == 1
    assert result.steps[0].attempts == 1


def test_steps_after_an_abort_still_count_as_skipped() -> None:
    class _Boom:
        kind = StepKind.RETRIEVE

        def run(self, context: StepContext) -> str:
            raise RuntimeError("boom")

    executors = {StepKind.RETRIEVE: _Boom(), StepKind.ANALYZE: _Stub(StepKind.ANALYZE)}
    result = ExecutionEngine(executors, abort_on_failure=True).run(
        _plan(StepKind.RETRIEVE, StepKind.ANALYZE)
    )
    assert (result.failed, result.skipped, result.aborted) == (1, 1, True)


def test_draft_pr_is_skipped_when_validation_never_ran(tmp_path: Path) -> None:
    executors = {
        StepKind.GENERATE: GenerateExecutor(_StubGenerator()),
        StepKind.DRAFT_PR: DraftPRExecutor(FakeDraftPRService(), owner="o", repo="r"),
    }
    result = ExecutionEngine(executors).run(_plan(StepKind.GENERATE, StepKind.DRAFT_PR))
    assert result.steps[0].status == StepStatus.COMPLETED
    assert result.steps[1].status == StepStatus.SKIPPED
    assert "validate the fix first" in result.steps[1].error


def test_full_chain_reaches_the_draft_pr(tmp_path: Path) -> None:
    pipeline = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    executors = {
        StepKind.GENERATE: GenerateExecutor(_StubGenerator()),
        StepKind.VALIDATE: ValidateExecutor(pipeline),
        StepKind.DRAFT_PR: DraftPRExecutor(FakeDraftPRService(), owner="o", repo="r"),
    }
    result = ExecutionEngine(executors).run(
        _plan(StepKind.GENERATE, StepKind.VALIDATE, StepKind.DRAFT_PR)
    )
    assert [s.status for s in result.steps] == [StepStatus.COMPLETED] * 3
    assert "Draft PR" in result.steps[2].output


def test_execute_tool_reports_skipped_count_and_reason(tmp_path: Path) -> None:
    tool = make_execute_plan_tool(RuleBasedPlanner(), _engine(tmp_path))
    out = tool.execute({"goal": "plan the refactor"})
    assert "Skipped:" in out
