"""
Typed models for the validation pipeline.

A ValidationResult is what a validated FixProposal gets back:
per-check pass/fail, a top-level score, and a decision on whether
the fix is safe to hand to the draft-PR stage.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CheckResult(BaseModel):
    """The outcome of running a single ValidationCheck."""

    model_config = {"frozen": True}

    name: str = Field(min_length=1, description="Human-readable check name.")
    passed: bool = Field(description="True when the check succeeded.")
    message: str = Field(default="", description="Detail line for humans / logs.")
    duration_ms: float = Field(ge=0.0, default=0.0, description="Wall-clock time in ms.")
    skipped: bool = Field(default=False, description="True when the check was intentionally skipped.")


class ValidationResult(BaseModel):
    """Aggregate result of running the whole ValidationPipeline."""

    model_config = {"frozen": True}

    proposal_goal: str = Field(min_length=1, description="Goal of the FixProposal that was validated.")
    checks: tuple = Field(default_factory=tuple, description="Per-check results.")
    passed: bool = Field(description="True when all non-skipped checks passed.")
    score: float = Field(ge=0.0, le=1.0, description="Fraction of non-skipped checks that passed.")
    error: str = Field(default="", description="Fatal error, if the pipeline aborted early.")

    @field_validator("score")
    @classmethod
    def _clamp_score(cls, v: float) -> float:
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    @property
    def failed_check_names(self) -> tuple:
        return tuple(c.name for c in self.checks if not c.passed and not c.skipped)
