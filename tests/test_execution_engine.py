"""Unit tests for backend.agent.engine (ExecutionEngine)."""

from __future__ import annotations

import pytest

from backend.agent.engine import ExecutionEngine, ExecutionResult, StepResult
from backend.agent.executor import StepContext
from backend.agent.models import Plan, PlanStep, StepKind, StepStatus


class _StubExecutor:
    """Configurable executor that always returns / raises what you tell it."""

    def __init__(self, kind: StepKind, output: str = "ok", raises: Exception | None = None) -> None:
        self.kind = kind
        self.output = output
        self.raises = raises
        self.call_count = 0

    def run(self, context: StepContext) -> str:
        self.call_count += 1
        if self.raises is not None:
            raise self.raises
        return self.output


def _plan(*kinds: StepKind, goal: str = "do the thing") -> Plan:
    steps = tuple(
        PlanStep(id=i + 1, kind=k, description=f"step {i + 1}")
        for i, k in enumerate(kinds)
    )
    return Plan(goal=goal, steps=steps, strategy="rule_based")


def test_engine_rejects_empty_executors() -> None:
    with pytest.raises(ValueError):
        ExecutionEngine(executors={})


def test_engine_rejects_zero_retries() -> None:
    with pytest.raises(ValueError):
        ExecutionEngine(executors={StepKind.RETRIEVE: _StubExecutor(StepKind.RETRIEVE)}, max_retries=0)


def test_engine_runs_all_steps() -> None:
    executors = {
        StepKind.RETRIEVE: _StubExecutor(StepKind.RETRIEVE, output="chunks"),
        StepKind.ANALYZE: _StubExecutor(StepKind.ANALYZE, output="answer"),
    }
    engine = ExecutionEngine(executors=executors)
    result = engine.run(_plan(StepKind.RETRIEVE, StepKind.ANALYZE))
    assert isinstance(result, ExecutionResult)
    assert result.total_steps == 2
    assert result.completed == 2
    assert result.failed == 0
    assert result.aborted is False
    assert all(s.status == StepStatus.COMPLETED for s in result.steps)


def test_engine_empty_plan_returns_zero_counts() -> None:
    executors = {StepKind.RETRIEVE: _StubExecutor(StepKind.RETRIEVE)}
    engine = ExecutionEngine(executors=executors)
    result = engine.run(Plan(goal="empty", steps=(), strategy="rule_based"))
    assert result.total_steps == 0
    assert result.completed == 0


def test_engine_missing_executor_marks_failed() -> None:
    executors = {StepKind.RETRIEVE: _StubExecutor(StepKind.RETRIEVE)}
    engine = ExecutionEngine(executors=executors)
    result = engine.run(_plan(StepKind.DRAFT_PR))
    assert result.failed == 1
    assert "No executor" in result.steps[0].error


def test_engine_retries_on_failure_then_succeeds() -> None:
    class Flaky:
        kind = StepKind.RETRIEVE

        def __init__(self) -> None:
            self.attempts = 0

        def run(self, context: StepContext) -> str:
            self.attempts += 1
            if self.attempts < 2:
                raise RuntimeError("transient")
            return "ok"

    engine = ExecutionEngine(executors={StepKind.RETRIEVE: Flaky()}, max_retries=2)
    result = engine.run(_plan(StepKind.RETRIEVE))
    assert result.completed == 1
    assert result.steps[0].attempts == 2


def test_engine_aborts_after_failure() -> None:
    executors = {
        StepKind.RETRIEVE: _StubExecutor(StepKind.RETRIEVE, raises=RuntimeError("boom")),
        StepKind.ANALYZE: _StubExecutor(StepKind.ANALYZE, output="won't run"),
    }
    engine = ExecutionEngine(executors=executors, max_retries=1, abort_on_failure=True)
    result = engine.run(_plan(StepKind.RETRIEVE, StepKind.ANALYZE))
    assert result.failed == 1
    assert result.aborted is True
    assert result.steps[1].status == StepStatus.SKIPPED


def test_engine_continues_when_abort_disabled() -> None:
    executors = {
        StepKind.RETRIEVE: _StubExecutor(StepKind.RETRIEVE, raises=RuntimeError("boom")),
        StepKind.ANALYZE: _StubExecutor(StepKind.ANALYZE, output="did run"),
    }
    engine = ExecutionEngine(executors=executors, abort_on_failure=False)
    result = engine.run(_plan(StepKind.RETRIEVE, StepKind.ANALYZE))
    assert result.failed == 1
    assert result.completed == 1
    assert result.aborted is False


def test_engine_passes_prior_outputs_to_later_steps() -> None:
    outputs_seen: list = []

    class Recorder:
        kind = StepKind.ANALYZE

        def run(self, context: StepContext) -> str:
            outputs_seen.append(dict(context.prior_outputs))
            return "done"

    executors = {
        StepKind.RETRIEVE: _StubExecutor(StepKind.RETRIEVE, output="chunk-A"),
        StepKind.ANALYZE: Recorder(),
    }
    engine = ExecutionEngine(executors=executors)
    engine.run(_plan(StepKind.RETRIEVE, StepKind.ANALYZE))
    assert outputs_seen[0] == {1: "chunk-A"}