"""
Typed models for the fix-generation layer.

A FixProposal is what the LLM produces + what our validator accepts.
A later stage will run validation against it; A later stage will convert it into
a draft PR.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, computed_field, field_validator


MAX_DIFF_CHARS = 200_000


def diff_fingerprint(diff: str) -> str:
    """SHA-256 of the exact diff text. The same diff always has the same fingerprint."""
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


class ReviewedDiffError(ValueError):
    """The reviewed diff cannot be used as sent. The message is safe to show."""


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

    @computed_field(description="SHA-256 of the diff text. Every stage shows the same value for the same diff.")  # type: ignore[prop-decorator]
    @property
    def diff_sha256(self) -> str:
        return diff_fingerprint(self.diff)

    @property
    def files_changed(self) -> int:
        return len(self.hunks)

    @property
    def total_added(self) -> int:
        return sum(h.added_lines for h in self.hunks)

    @property
    def total_removed(self) -> int:
        return sum(h.removed_lines for h in self.hunks)


class ReviewedProposal(BaseModel):
    """A proposal the user has already reviewed, sent back by the client.

    The server uses this diff as is. It never asks the AI for a new one.
    """

    goal: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    diff: str = Field(min_length=1, max_length=MAX_DIFF_CHARS)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    model: str = Field(default="reviewed", min_length=1)
    diff_sha256: str | None = Field(
        default=None,
        description="Optional: the fingerprint shown when the diff was reviewed. A mismatch is refused.",
    )
