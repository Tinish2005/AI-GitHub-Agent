"""Unit tests for backend.agent.draft_pr_models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.agent.draft_pr_models import DraftPRRequest, DraftPRResult


def _req(**overrides) -> DraftPRRequest:
    defaults = dict(
        owner="Tinish2005",
        repo="AI-GitHub-Agent",
        base_branch="main",
        goal="Add greet helper",
        proposal_explanation="Adds a greet() helper.",
        proposal_diff="--- /dev/null\n+++ b/hello.py\n@@ -0,0 +1,1 @@\n+greet\n",
        confidence=0.8,
        validation_passed=True,
        validation_score=1.0,
    )
    defaults.update(overrides)
    return DraftPRRequest(**defaults)


def test_request_defaults_and_shape() -> None:
    r = _req()
    assert r.base_branch == "main"
    assert r.confidence == 0.8
    assert r.validation_score == 1.0


def test_request_rejects_empty_owner() -> None:
    with pytest.raises(ValidationError):
        _req(owner="")


def test_request_rejects_empty_repo() -> None:
    with pytest.raises(ValidationError):
        _req(repo="")


def test_request_rejects_empty_goal() -> None:
    with pytest.raises(ValidationError):
        _req(goal="")


def test_request_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError):
        _req(confidence=1.5)


def test_request_is_frozen() -> None:
    r = _req()
    with pytest.raises(ValidationError):
        r.owner = "x"  # type: ignore[misc]


def test_result_defaults_when_not_created() -> None:
    r = DraftPRResult(created=False)
    assert r.pr_number == 0
    assert r.pr_url == ""
    assert r.branch == ""


def test_result_when_created() -> None:
    r = DraftPRResult(
        created=True,
        pr_number=42,
        pr_url="https://github.com/o/r/pull/42",
        branch="agent/fix",
        title="[agent] fix",
        body="details",
    )
    assert r.pr_number == 42