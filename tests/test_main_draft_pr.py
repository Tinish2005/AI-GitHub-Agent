"""Integration tests for the /fix/pr endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.agent.draft_pr_service import FakeDraftPRService
from backend.agent.fix_generator import FixGenerator
from backend.agent.validation_checks import SyntaxCheck
from backend.agent.validation_pipeline import ValidationPipeline
from backend.main import (
    app, get_draft_pr_service, get_fix_generator, get_validation_pipeline,
)


class _StaticLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    @property
    def model(self) -> str:
        return "static-fake"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


# Valid two-line diff.
VALID_DIFF = (
    "--- /dev/null\n"
    "+++ b/hello.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+def greet():\n"
    "+    return 'hi'\n"
)


def _valid_response() -> str:
    return json.dumps({"explanation": "add greet", "diff": VALID_DIFF, "confidence": 0.9})


@pytest.fixture
def client(tmp_path: Path):
    gen = FixGenerator(llm=_StaticLLM(_valid_response()))
    validator = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    svc = FakeDraftPRService(next_number=300)
    app.dependency_overrides[get_fix_generator] = lambda: gen
    app.dependency_overrides[get_validation_pipeline] = lambda: validator
    app.dependency_overrides[get_draft_pr_service] = lambda: svc
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_pr_endpoint_creates_draft(client: TestClient) -> None:
    resp = client.post("/fix/pr", json={
        "goal": "add greet", "context": "",
        "owner": "Tinish2005", "repo": "AI-GitHub-Agent",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["created"] is True
    assert body["pr_number"] == 300
    assert "pull/300" in body["pr_url"]


def test_pr_endpoint_rejects_empty_goal(client: TestClient) -> None:
    resp = client.post("/fix/pr", json={
        "goal": "", "context": "",
        "owner": "Tinish2005", "repo": "AI-GitHub-Agent",
    })
    assert resp.status_code in (400, 422)


def test_pr_endpoint_reports_skip_when_validation_fails(tmp_path: Path) -> None:
    bad = json.dumps({"explanation": "e", "diff": "not a diff"})
    gen = FixGenerator(llm=_StaticLLM(bad))
    validator = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    svc = FakeDraftPRService()
    app.dependency_overrides[get_fix_generator] = lambda: gen
    app.dependency_overrides[get_validation_pipeline] = lambda: validator
    app.dependency_overrides[get_draft_pr_service] = lambda: svc
    try:
        client = TestClient(app)
        resp = client.post("/fix/pr", json={
            "goal": "add greet", "context": "",
            "owner": "Tinish2005", "repo": "AI-GitHub-Agent",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["created"] is False
    finally:
        app.dependency_overrides.clear()