"""Unit tests for backend.agent.fix_models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.agent.fix_models import FixHunk, FixProposal


def _hunk(added: int = 3, removed: int = 1) -> FixHunk:
    return FixHunk(
        file_path="src/app.py", added_lines=added, removed_lines=removed,
    )


def test_hunk_defaults() -> None:
    h = _hunk()
    assert h.is_new_file is False
    assert h.is_deleted_file is False


def test_hunk_rejects_empty_path() -> None:
    with pytest.raises(ValidationError):
        FixHunk(file_path="", added_lines=0, removed_lines=0)


def test_hunk_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        FixHunk(file_path="a.py", added_lines=-1, removed_lines=0)


def test_proposal_computes_totals() -> None:
    p = FixProposal(
        goal="fix stuff",
        explanation="do the thing",
        diff="--- a\n+++ b\n",
        hunks=(_hunk(3, 1), _hunk(2, 4)),
        model="fake",
        confidence=0.7,
    )
    assert p.files_changed == 2
    assert p.total_added == 5
    assert p.total_removed == 5


def test_proposal_confidence_clamped_low() -> None:
    with pytest.raises(ValidationError):
        FixProposal(
            goal="g", explanation="e", diff="d",
            model="fake", confidence=-0.5,
        )


def test_proposal_confidence_clamped_high() -> None:
    with pytest.raises(ValidationError):
        FixProposal(
            goal="g", explanation="e", diff="d",
            model="fake", confidence=1.2,
        )


def test_proposal_is_valid_default_true() -> None:
    p = FixProposal(
        goal="g", explanation="e", diff="d", model="fake", confidence=0.5,
    )
    assert p.is_valid is True
    assert p.validation_error == ""


def test_proposal_is_frozen() -> None:
    p = FixProposal(
        goal="g", explanation="e", diff="d", model="fake", confidence=0.5,
    )
    with pytest.raises(ValidationError):
        p.goal = "other"  # type: ignore[misc]