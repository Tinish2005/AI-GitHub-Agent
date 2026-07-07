"""Integration tests for the /fix/validate endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.agent.fix_generator import FixGenerator
from backend.agent.validation_checks import SyntaxCheck
from backend.agent.validation_pipeline import ValidationPipeline
from backend.main import app, get_fix_generator, get_validation_pipeline


class _StaticLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    @property
    def model(self) -> str:
        return "static-fake"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


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
    app.dependency_overrides[get_fix_generator] = lambda: gen
    app.dependency_overrides[get_validation_pipeline] = lambda: validator
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_validate_endpoint_returns_200(client: TestClient) -> None:
    resp = client.post("/fix/validate", json={"goal": "add greet", "context": ""})
    assert resp.status_code == 200
    body = resp.json()
    assert body["proposal_goal"] == "add greet"
    assert body["passed"] is True
    assert body["score"] == 1.0
    names = [c["name"] for c in body["checks"]]
    assert "syntax" in names


def test_validate_endpoint_rejects_empty_goal(client: TestClient) -> None:
    resp = client.post("/fix/validate", json={"goal": "", "context": ""})
    assert resp.status_code in (400, 422)


def test_validate_endpoint_reports_invalid_proposal(tmp_path: Path) -> None:
    bad = json.dumps({"explanation": "e", "diff": "not a diff at all"})
    gen = FixGenerator(llm=_StaticLLM(bad))
    validator = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    app.dependency_overrides[get_fix_generator] = lambda: gen
    app.dependency_overrides[get_validation_pipeline] = lambda: validator
    try:
        client = TestClient(app)
        resp = client.post("/fix/validate", json={"goal": "x", "context": ""})
        assert resp.status_code == 200
        body = resp.json()
        assert body["passed"] is False
        assert body["error"]
    finally:
        app.dependency_overrides.clear()