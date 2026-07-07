"""Unit tests for backend.agent.validation_models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.agent.validation_models import CheckResult, ValidationResult


def test_check_result_defaults() -> None:
    c = CheckResult(name="syntax", passed=True)
    assert c.message == ""
    assert c.skipped is False
    assert c.duration_ms == 0.0


def test_check_result_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        CheckResult(name="", passed=True)


def test_check_result_rejects_negative_duration() -> None:
    with pytest.raises(ValidationError):
        CheckResult(name="s", passed=True, duration_ms=-1.0)


def test_validation_result_totals() -> None:
    r = ValidationResult(
        proposal_goal="g",
        checks=(
            CheckResult(name="syntax", passed=True),
            CheckResult(name="imports", passed=False, message="bad"),
            CheckResult(name="lint", passed=True, skipped=True),
        ),
        passed=False,
        score=0.5,
    )
    assert r.total_checks == 3
    assert r.failed_check_names == ("imports",)


def test_validation_result_score_bounds_low() -> None:
    with pytest.raises(ValidationError):
        ValidationResult(proposal_goal="g", checks=(), passed=True, score=-0.1)


def test_validation_result_score_bounds_high() -> None:
    with pytest.raises(ValidationError):
        ValidationResult(proposal_goal="g", checks=(), passed=True, score=1.5)


def test_validation_result_is_frozen() -> None:
    r = ValidationResult(proposal_goal="g", checks=(), passed=True, score=1.0)
    with pytest.raises(ValidationError):
        r.passed = False  # type: ignore[misc]