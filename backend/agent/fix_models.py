"""
Typed models for the fix-generation layer.

A FixProposal is what the LLM produces + what our validator accepts.
A later stage will run validation against it; A later stage will convert it into
a draft PR.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class FixHunk(BaseModel):
    """One file change inside a proposal (mirrors a unified-diff file entry)."""

    model_config = {"frozen": True}

    file_path: str = Field(min_length=1, description="Path to the file being changed.")
    is_new_file: bool = Field(default=False, description="True if the diff creates the file.")
    is_deleted_file: bool = Field(default=False, description="True if the diff deletes the file.")
    added_lines: int = Field(ge=0, description="Total lines added in this hunk.")
    removed_lines: int = Field(ge=0, description="Total lines removed in this hunk.")


class FixProposal(BaseModel):
    """A proposed fix returned by the FixGenerator."""

    model_config = {"frozen": True}

    goal: str = Field(min_length=1, description="The user goal that produced this proposal.")
    explanation: str = Field(min_length=1, description="Plain-English rationale for the fix.")
    diff: str = Field(min_length=1, description="Full unified-diff text.")
    hunks: tuple = Field(default_factory=tuple, description="Per-file change summary.")
    model: str = Field(min_length=1, description="LLM model identifier used.")
    confidence: float = Field(ge=0.0, le=1.0, description="Model / heuristic confidence in the fix.")
    is_valid: bool = Field(default=True, description="True if the diff parsed cleanly.")
    validation_error: str = Field(default="", description="Reason the diff was rejected, if any.")

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        # Belt and braces on top of the ge/le on Field.
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    @property
    def files_changed(self) -> int:
        return len(self.hunks)

    @property
    def total_added(self) -> int:
        return sum(h.added_lines for h in self.hunks)

    @property
    def total_removed(self) -> int:
        return sum(h.removed_lines for h in self.hunks)