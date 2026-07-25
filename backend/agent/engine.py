"""
Execution engine.

Takes a Plan and runs each step in order. For each step:
    - Picks the matching StepExecutor by kind
    - Calls executor.run(context) with the accumulated prior outputs
    - Captures the result, marks the step COMPLETED
    - On failure, retries once; if still failing, marks FAILED and
      either continues or aborts depending on the abort_on_failure flag

Returns a typed ExecutionResult with per-step results and a top-level
status. Future work can add replanning; today we ship the "run + report"
core.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.agent.executor import StepContext, StepExecutor
from backend.agent.models import Plan, PlanStep, StepKind, StepStatus


class StepResult(BaseModel):
    """The outcome of running a single PlanStep."""

    model_config = {"frozen": True}

    step_id: int = Field(ge=1)
    kind: StepKind
    status: StepStatus
    output: str = Field(default="")
    error: str = Field(default="")
    attempts: int = Field(ge=1, default=1)


class ExecutionResult(BaseModel):
    """The outcome of running a whole Plan."""

    model_config = {"frozen": True}

    goal: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    steps: tuple = Field(default_factory=tuple)
    completed: int = Field(ge=0, default=0)
    failed: int = Field(ge=0, default=0)
    aborted: bool = Field(default=False)

    @property
    def total_steps(self) -> int:
        return len(self.steps)


class ExecutionEngine:
    """Runs a Plan step by step using a StepKind -> StepExecutor map."""

    def __init__(
        self,
        executors: dict,
        *,
        max_retries: int = 1,
        abort_on_failure: bool = True,
    ) -> None:
        if not executors:
            raise ValueError("At least one executor must be registered.")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1.")
        self.executors = executors
        self.max_retries = max_retries
        self.abort_on_failure = abort_on_failure

    def run(self, plan: Plan) -> ExecutionResult:
        """Execute the whole plan; return a structured result."""
        if plan.step_count == 0:
            return ExecutionResult(
                goal=plan.goal, strategy=plan.strategy, steps=(), completed=0, failed=0,
            )

        results: list = []
        prior_outputs: dict = {}
        aborted = False

        for step in plan.steps:
            if aborted:
                results.append(
                    StepResult(
                        step_id=step.id, kind=step.kind,
                        status=StepStatus.SKIPPED, output="", error="aborted by earlier failure",
                    )
                )
                continue

            result = self._run_step(plan.goal, step, prior_outputs)
            results.append(result)
            prior_outputs[step.id] = result.output

            if result.status == StepStatus.FAILED and self.abort_on_failure:
                aborted = True

        completed = sum(1 for r in results if r.status == StepStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == StepStatus.FAILED)

        return ExecutionResult(
            goal=plan.goal,
            strategy=plan.strategy,
            steps=tuple(results),
            completed=completed,
            failed=failed,
            aborted=aborted,
        )

    def _run_step(
        self,
        goal: str,
        step: PlanStep,
        prior_outputs: dict,
    ) -> StepResult:
        executor = self.executors.get(step.kind)
        if executor is None:
            return StepResult(
                step_id=step.id, kind=step.kind, status=StepStatus.FAILED,
                error=f"No executor registered for kind '{step.kind.value}'.",
                attempts=1,
            )

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                context = StepContext(goal=goal, step=step, prior_outputs=prior_outputs)
                output = executor.run(context)
                return StepResult(
                    step_id=step.id, kind=step.kind, status=StepStatus.COMPLETED,
                    output=output, attempts=attempt,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        return StepResult(
            step_id=step.id, kind=step.kind, status=StepStatus.FAILED,
            error=last_error, attempts=self.max_retries,
        )