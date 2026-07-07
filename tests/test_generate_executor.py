"""Unit tests for GenerateExecutor."""

from __future__ import annotations

import json

from backend.agent.executor import GenerateExecutor, StepContext
from backend.agent.fix_generator import FixGenerator
from backend.agent.models import PlanStep, StepKind


class _StaticLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    @property
    def model(self) -> str:
        return "static-fake"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


VALID_DIFF = (
    "--- a/src/app.py\n"
    "+++ b/src/app.py\n"
    "@@ -1,1 +1,1 @@\n"
    "-return a - b\n"
    "+return a + b\n"
)


def _valid_response() -> str:
    return json.dumps({
        "explanation": "swap - for +",
        "diff": VALID_DIFF,
        "confidence": 0.8,
    })


def _step(kind: StepKind = StepKind.GENERATE) -> PlanStep:
    return PlanStep(id=4, kind=kind, description="generate the fix")


def test_generate_executor_returns_summary_text() -> None:
    gen = FixGenerator(llm=_StaticLLM(_valid_response()))
    ex = GenerateExecutor(gen)
    ctx = StepContext(goal="fix add", step=_step(), prior_outputs={1: "some retrieval context"})
    out = ex.run(ctx)
    assert "Fix proposal" in out
    assert "confidence=0.80" in out
    assert "is_valid=True" in out


def test_generate_executor_uses_retrieve_output_as_context() -> None:
    captured: dict = {}

    class SpyGen:
        model = "spy"

        def propose(self, goal: str, context: str):
            captured["goal"] = goal
            captured["context"] = context
            from backend.agent.fix_models import FixProposal
            return FixProposal(
                goal=goal, explanation="ok", diff="d",
                model=self.model, confidence=0.5,
            )

    ex = GenerateExecutor(SpyGen())  # type: ignore[arg-type]
    ctx = StepContext(
        goal="my goal",
        step=_step(),
        prior_outputs={1: "retrieval-output-1", 2: "analyze-output-2"},
    )
    ex.run(ctx)
    assert captured["goal"] == "my goal"
    assert captured["context"] == "retrieval-output-1"


def test_generate_executor_reports_invalid_diff() -> None:
    bad = json.dumps({"explanation": "nope", "diff": "not a diff"})
    gen = FixGenerator(llm=_StaticLLM(bad))
    ex = GenerateExecutor(gen)
    ctx = StepContext(goal="fix add", step=_step(), prior_outputs={})
    out = ex.run(ctx)
    assert "is_valid=False" in out
    assert "Validation error" in out