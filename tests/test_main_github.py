"""Integration tests for the /github/file endpoint."""

from __future__ import annotations

from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.github.client import FakeGitHubClient
from backend.github.models import RepoCoord
from backend.main import app, get_github_client


@pytest.fixture
def fake() -> FakeGitHubClient:
    c = FakeGitHubClient()
    c.add_file(
        RepoCoord(owner="Tinish2005", repo="AI-GitHub-Agent"),
        "README.md",
        "Hello from tests",
    )
    return c


@pytest.fixture
def client(fake: FakeGitHubClient) -> Iterator[TestClient]:
    app.dependency_overrides[get_github_client] = lambda: fake
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_github_file_endpoint_returns_200(client: TestClient) -> None:
    resp = client.get(
        "/github/file",
        params={"owner": "Tinish2005", "repo": "AI-GitHub-Agent", "path": "README.md"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "Hello from tests"
    assert body["path"] == "README.md"


def test_github_file_endpoint_404_for_missing(client: TestClient) -> None:
    resp = client.get(
        "/github/file",
        params={"owner": "Tinish2005", "repo": "AI-GitHub-Agent", "path": "missing.md"},
    )
    assert resp.status_code == 404


def test_github_file_endpoint_400_for_missing_params(client: TestClient) -> None:
    resp = client.get("/github/file", params={"owner": "x", "repo": "y"})
    assert resp.status_code == 422