"""Integration tests for the GitHub MCP tools."""

from __future__ import annotations

import pytest

from backend.github.client import FakeGitHubClient
from backend.github.models import Issue, PullRequest, RepoCoord
from backend.mcp.tools import (
    make_github_get_file_tool,
    make_github_get_pr_tool,
    make_github_list_issues_tool,
)


@pytest.fixture
def coord() -> RepoCoord:
    return RepoCoord(owner="Tinish2005", repo="AI-GitHub-Agent")


@pytest.fixture
def fake(coord: RepoCoord) -> FakeGitHubClient:
    c = FakeGitHubClient()
    c.add_file(coord, "README.md", "Hello agent")
    c.add_issue(coord, Issue(
        number=1, title="First bug", state="open",
        author="tinish", url="u",
    ))
    c.add_pr(coord, PullRequest(
        number=7, title="Cool PR", state="open",
        author="tinish", head="feat/x", base="main", url="up",
        body="This is a great PR.",
    ))
    return c


def test_get_file_tool_returns_content(fake: FakeGitHubClient) -> None:
    tool = make_github_get_file_tool(fake)
    params = {"owner": "Tinish2005", "repo": "AI-GitHub-Agent", "path": "README.md"}
    out = tool.execute(params)
    assert "Hello agent" in out
    assert "README.md" in out


def test_get_file_tool_requires_params(fake: FakeGitHubClient) -> None:
    tool = make_github_get_file_tool(fake)
    with pytest.raises(ValueError):
        tool.execute({"owner": "", "repo": "r", "path": "p"})
    with pytest.raises(ValueError):
        tool.execute({"owner": "o", "repo": "", "path": "p"})
    with pytest.raises(ValueError):
        tool.execute({"owner": "o", "repo": "r", "path": ""})


def test_get_file_tool_missing_file(fake: FakeGitHubClient) -> None:
    tool = make_github_get_file_tool(fake)
    params = {"owner": "Tinish2005", "repo": "AI-GitHub-Agent", "path": "does-not-exist.txt"}
    with pytest.raises(ValueError):
        tool.execute(params)


def test_list_issues_tool_returns_open(fake: FakeGitHubClient) -> None:
    tool = make_github_list_issues_tool(fake)
    params = {"owner": "Tinish2005", "repo": "AI-GitHub-Agent"}
    out = tool.execute(params)
    assert "First bug" in out
    assert "open" in out.lower()


def test_list_issues_tool_empty(fake: FakeGitHubClient) -> None:
    tool = make_github_list_issues_tool(fake)
    params = {"owner": "Tinish2005", "repo": "AI-GitHub-Agent", "state": "closed"}
    out = tool.execute(params)
    assert "No closed issues" in out


def test_list_issues_tool_validates_state(fake: FakeGitHubClient) -> None:
    tool = make_github_list_issues_tool(fake)
    with pytest.raises(ValueError):
        tool.execute({"owner": "o", "repo": "r", "state": "weird"})


def test_list_issues_tool_validates_per_page(fake: FakeGitHubClient) -> None:
    tool = make_github_list_issues_tool(fake)
    with pytest.raises(ValueError):
        tool.execute({"owner": "o", "repo": "r", "per_page": 0})
    with pytest.raises(ValueError):
        tool.execute({"owner": "o", "repo": "r", "per_page": 101})


def test_get_pr_tool_returns_pr_info(fake: FakeGitHubClient) -> None:
    tool = make_github_get_pr_tool(fake)
    params = {"owner": "Tinish2005", "repo": "AI-GitHub-Agent", "number": 7}
    out = tool.execute(params)
    assert "Cool PR" in out
    assert "feat/x" in out
    assert "main" in out


def test_get_pr_tool_requires_params(fake: FakeGitHubClient) -> None:
    tool = make_github_get_pr_tool(fake)
    with pytest.raises(ValueError):
        tool.execute({"owner": "", "repo": "r", "number": 1})
    with pytest.raises(ValueError):
        tool.execute({"owner": "o", "repo": "r", "number": 0})


def test_get_pr_tool_missing(fake: FakeGitHubClient) -> None:
    tool = make_github_get_pr_tool(fake)
    params = {"owner": "Tinish2005", "repo": "AI-GitHub-Agent", "number": 999}
    with pytest.raises(ValueError):
        tool.execute(params)