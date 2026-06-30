"""Unit tests for backend.github.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.github.models import GitHubFile, Issue, PullRequest, RepoCoord


def test_repo_coord_slug() -> None:
    coord = RepoCoord(owner="Tinish2005", repo="AI-GitHub-Agent")
    assert coord.slug() == "Tinish2005/AI-GitHub-Agent"


def test_repo_coord_is_frozen() -> None:
    coord = RepoCoord(owner="o", repo="r")
    with pytest.raises(ValidationError):
        coord.owner = "x"  # type: ignore[misc]


def test_repo_coord_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        RepoCoord(owner="", repo="r")
    with pytest.raises(ValidationError):
        RepoCoord(owner="o", repo="")


def test_github_file_defaults_and_validation() -> None:
    f = GitHubFile(path="README.md", sha="abc", size=10, content="hi")
    assert f.encoding == "base64"
    assert f.size == 10
    with pytest.raises(ValidationError):
        GitHubFile(path="", sha="a", size=0, content="x")


def test_issue_minimal_construction() -> None:
    i = Issue(
        number=1,
        title="Bug",
        state="open",
        author="tinish",
        url="https://github.com/o/r/issues/1",
    )
    assert i.body == ""
    assert i.state == "open"


def test_issue_rejects_invalid_number() -> None:
    with pytest.raises(ValidationError):
        Issue(number=0, title="x", state="open", author="a", url="u")


def test_pull_request_construction() -> None:
    pr = PullRequest(
        number=5,
        title="Fix bug",
        state="open",
        author="tinish",
        head="feature/x",
        base="main",
        url="https://github.com/o/r/pull/5",
    )
    assert pr.merged is False
    assert pr.body == ""


def test_pull_request_merged_flag() -> None:
    pr = PullRequest(
        number=10,
        title="Done",
        state="closed",
        author="t",
        head="b",
        base="main",
        url="u",
        merged=True,
    )
    assert pr.merged is True