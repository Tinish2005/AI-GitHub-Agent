"""
Typed models for the draft-PR layer.

DraftPRRequest is what the caller sends in. DraftPRResult is what the
service returns after successfully creating (or simulating) a draft PR.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class DraftPRRequest(BaseModel):
    """A request to create a draft PR from a validated FixProposal."""

    model_config = {"frozen": True}

    owner: str = Field(min_length=1, description="GitHub owner or org login.")
    repo: str = Field(min_length=1, description="Repository name.")
    base_branch: str = Field(default="main", min_length=1, description="Target branch to merge into.")
    goal: str = Field(min_length=1, description="Original user goal (also used to derive names).")
    proposal_explanation: str = Field(min_length=1, description="Human-readable explanation for the PR body.")
    proposal_diff: str = Field(min_length=1, description="Unified diff text to publish.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the fix.")
    validation_passed: bool = Field(description="True if the validation pipeline said the fix is safe.")
    validation_score: float = Field(ge=0.0, le=1.0, description="Validation score (0.0 to 1.0).")

    @field_validator("validation_score", "confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v


class DraftPRResult(BaseModel):
    """The outcome of publishing a draft PR."""

    model_config = {"frozen": True}

    created: bool = Field(description="True if the draft PR was published successfully.")
    pr_number: int = Field(ge=0, default=0, description="The PR number (0 if not created).")
    pr_url: str = Field(default="", description="Public URL of the PR (empty if not created).")
    branch: str = Field(default="", description="The head branch used for the PR.")
    title: str = Field(default="", description="Title used for the PR.")
    body: str = Field(default="", description="Body used for the PR.")
    error: str = Field(default="", description="Reason the PR was NOT created, if any.")
    skipped_reason: str = Field(default="", description="Reason we deliberately skipped creating the PR.")