"""
Data models for the agent planning layer.

A `Plan` is an ordered list of `PlanStep`s. Each step declares a `kind`
(what category of work it is) and a plain-English `description`. Steps
are pure data - the execution engine (Loop 10) will actually run them.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class StepKind(str, Enum):
    """Category of work a plan step represents."""

    RETRIEVE = "retrieve"          # Search or fetch code / context
    ANALYZE = "analyze"            # Reason over retrieved context
    GITHUB_READ = "github_read"    # Read from GitHub (files, issues, PRs)
    GENERATE = "generate"          # Generate a code diff or answer
    VALIDATE = "validate"          # Run build / tests / lint / security
    HUMAN_APPROVAL = "human_approval"  # Wait for a human decision
    DRAFT_PR = "draft_pr"          # Push a draft pull request


class StepStatus(str, Enum):
    """Runtime status of a plan step (set by the executor, not the planner)."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    """A single step in a plan."""

    model_config = {"frozen": True}

    id: int = Field(ge=1, description="1-indexed position of the step.")
    kind: StepKind = Field(description="Category of work.")
    description: str = Field(min_length=1, description="Plain-English description.")
    depends_on: tuple = Field(
        default_factory=tuple,
        description="IDs of steps that must complete before this one.",
    )
    status: StepStatus = Field(default=StepStatus.PENDING)


class Plan(BaseModel):
    """A structured, multi-step execution plan produced by a Planner."""

    model_config = {"frozen": True}

    goal: str = Field(min_length=1, description="The user goal this plan addresses.")
    steps: tuple = Field(default_factory=tuple, description="Ordered plan steps.")
    strategy: str = Field(
        default="rule_based",
        description="Which planner produced this plan (rule_based / llm).",
    )

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def kinds(self) -> tuple:
        """Return the ordered tuple of step kinds in the plan."""
        return tuple(s.kind for s in self.steps)