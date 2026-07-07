"""Unit tests for FakeDraftPRService + branch/title/body helpers."""

from __future__ import annotations

import pytest

from backend.agent.draft_pr_models import DraftPRRequest
from backend.agent.draft_pr_service import (
    FakeDraftPRService, GitHubDraftPRService,
    _build_branch, _build_body, _build_title,
    _extract_new_files_from_diff,
)


VALID_DIFF = (
    "--- /dev/null\n"
    "+++ b/hello.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+def greet():\n"
    "+    return 'hi'\n"
)


def _req(**overrides) -> DraftPRRequest:
    defaults = dict(
        owner="Tinish2005",
        repo="AI-GitHub-Agent",
        base_branch="main",
        goal="Add greet helper",
        proposal_explanation="Adds a greet() helper.",
        proposal_diff=VALID_DIFF,
        confidence=0.9,
        validation_passed=True,
        validation_score=1.0,
    )
    defaults.update(overrides)
    return DraftPRRequest(**defaults)


def test_branch_is_slugified() -> None:
    branch = _build_branch("Fix a Login Bug!!")
    assert branch.startswith("agent/fix-a-login-bug")


def test_branch_falls_back_when_goal_empty_of_letters() -> None:
    branch = _build_branch("!!!!")
    assert branch.startswith("agent/agent-fix")


def test_title_is_prefixed() -> None:
    assert _build_title("do the thing").startswith("[agent] ")


def test_body_includes_diff_and_validation() -> None:
    body = _build_body(_req())
    assert "### Goal" in body
    assert "### Validation" in body
    assert "def greet" in body
    assert "Human review required" in body


def test_extract_new_files_parses_added_lines() -> None:
    files = _extract_new_files_from_diff(VALID_DIFF)
    assert list(files.keys()) == ["hello.py"]
    assert files["hello.py"].startswith("def greet():")


def test_extract_new_files_ignores_non_new_file_diffs() -> None:
    modify_diff = (
        "--- a/existing.py\n"
        "+++ b/existing.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    )
    assert _extract_new_files_from_diff(modify_diff) == {}


def test_fake_service_creates_pr_when_valid() -> None:
    svc = FakeDraftPRService(next_number=100)
    result = svc.create(_req())
    assert result.created is True
    assert result.pr_number == 100
    assert result.pr_url.endswith("/pull/100")
    assert "hello" not in result.title  # goal is "Add greet helper"
    assert svc.calls[0].goal == "Add greet helper"


def test_fake_service_refuses_when_validation_failed() -> None:
    svc = FakeDraftPRService()
    result = svc.create(_req(validation_passed=False, validation_score=0.3))
    assert result.created is False
    assert "Validation did not pass" in result.skipped_reason


def test_fake_service_rejects_empty_diff() -> None:
    svc = FakeDraftPRService()
    result = svc.create(_req(proposal_diff="   "))
    assert result.created is False
    assert "empty" in result.error.lower()


def test_fake_service_increments_pr_number() -> None:
    svc = FakeDraftPRService(next_number=5)
    a = svc.create(_req())
    b = svc.create(_req())
    assert a.pr_number == 5
    assert b.pr_number == 6


def test_real_service_rejects_empty_token() -> None:
    with pytest.raises(ValueError):
        GitHubDraftPRService(token="")


def test_real_service_headers_include_token() -> None:
    svc = GitHubDraftPRService(token="ghp_fake")
    headers = svc._headers()
    assert headers["Authorization"] == "Bearer ghp_fake"
    assert headers["Accept"] == "application/vnd.github+json"