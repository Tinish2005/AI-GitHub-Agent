"""Unit tests for backend.github.client (using the fake backend)."""

from __future__ import annotations

import pytest

from backend.github.client import FakeGitHubClient, GitHubClient
from backend.github.models import Issue, PullRequest, RepoCoord


@pytest.fixture
def coord() -> RepoCoord:
    return RepoCoord(owner="Tinish2005", repo="AI-GitHub-Agent")


@pytest.fixture
def fake(coord: RepoCoord) -> FakeGitHubClient:
    c = FakeGitHubClient()
    c.add_file(coord, "README.md", "Hello world")
    c.add_file(coord, "src/main.py", "print('hi')", ref="develop")
    c.add_issue(coord, Issue(
        number=1, title="Bug A", state="open",
        author="alice", url="u1",
    ))
    c.add_issue(coord, Issue(
        number=2, title="Bug B", state="closed",
        author="bob", url="u2",
    ))
    c.add_pr(coord, PullRequest(
        number=42, title="Feature X", state="open",
        author="alice", head="feature/x", base="main", url="up",
    ))
    return c


def test_fake_get_file(fake: FakeGitHubClient, coord: RepoCoord) -> None:
    f = fake.get_file(coord, "README.md")
    assert f.content == "Hello world"
    assert f.size == len("Hello world".encode("utf-8"))


def test_fake_get_file_with_ref(fake: FakeGitHubClient, coord: RepoCoord) -> None:
    f = fake.get_file(coord, "src/main.py", ref="develop")
    assert "print" in f.content


def test_fake_get_file_missing(fake: FakeGitHubClient, coord: RepoCoord) -> None:
    with pytest.raises(FileNotFoundError):
        fake.get_file(coord, "nope.md")


def test_fake_list_issues_open_only(fake: FakeGitHubClient, coord: RepoCoord) -> None:
    issues = fake.list_issues(coord, state="open")
    assert len(issues) == 1
    assert issues[0].number == 1


def test_fake_list_issues_all(fake: FakeGitHubClient, coord: RepoCoord) -> None:
    issues = fake.list_issues(coord, state="all")
    assert len(issues) == 2


def test_fake_list_issues_pagination(fake: FakeGitHubClient, coord: RepoCoord) -> None:
    issues = fake.list_issues(coord, state="all", per_page=1)
    assert len(issues) == 1


def test_fake_get_pr(fake: FakeGitHubClient, coord: RepoCoord) -> None:
    pr = fake.get_pr(coord, 42)
    assert pr.title == "Feature X"
    assert pr.head == "feature/x"


def test_fake_get_pr_missing(fake: FakeGitHubClient, coord: RepoCoord) -> None:
    with pytest.raises(FileNotFoundError):
        fake.get_pr(coord, 999)


def test_real_client_rejects_invalid_state(coord: RepoCoord) -> None:
    client = GitHubClient()
    with pytest.raises(ValueError):
        client.list_issues(coord, state="weird")


def test_real_client_rejects_bad_per_page(coord: RepoCoord) -> None:
    client = GitHubClient()
    with pytest.raises(ValueError):
        client.list_issues(coord, per_page=0)
    with pytest.raises(ValueError):
        client.list_issues(coord, per_page=101)


def test_real_client_rejects_empty_path(coord: RepoCoord) -> None:
    client = GitHubClient()
    with pytest.raises(ValueError):
        client.get_file(coord, "")


def test_real_client_rejects_bad_pr_number(coord: RepoCoord) -> None:
    client = GitHubClient()
    with pytest.raises(ValueError):
        client.get_pr(coord, 0)


def test_real_client_headers_without_token() -> None:
    client = GitHubClient(token=None)
    headers = client._headers()
    assert "Authorization" not in headers
    assert headers["Accept"] == "application/vnd.github+json"


def test_real_client_headers_with_token() -> None:
    client = GitHubClient(token="ghp_fake")
    headers = client._headers()
    assert headers["Authorization"] == "Bearer ghp_fake"