"""Integration tests for the /fix/propose endpoint."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.agent.fix_generator import FixGenerator
from backend.main import app, get_fix_generator


class _StaticLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    @property
    def model(self) -> str:
        return "static-fake"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


VALID_DIFF = (
    "--- a/src/app.py\n"
    "+++ b/src/app.py\n"
    "@@ -1,1 +1,1 @@\n"
    "-return a - b\n"
    "+return a + b\n"
)


def _valid_response() -> str:
    return json.dumps({
        "explanation": "swap - for +",
        "diff": VALID_DIFF,
        "confidence": 0.8,
    })


@pytest.fixture
def client():
    gen = FixGenerator(llm=_StaticLLM(_valid_response()))
    app.dependency_overrides[get_fix_generator] = lambda: gen
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_fix_propose_returns_200(client: TestClient) -> None:
    resp = client.post("/fix/propose", json={"goal": "fix add", "context": ""})
    assert resp.status_code == 200
    body = resp.json()
    assert body["goal"] == "fix add"
    assert body["model"] == "static-fake"
    assert body["is_valid"] is True
    # `hunks` is the serialized field (Pydantic tuple field);
    # `files_changed` is a @property so it's NOT in the JSON payload.
    assert len(body["hunks"]) >= 1


def test_fix_propose_rejects_empty_goal(client: TestClient) -> None:
    resp = client.post("/fix/propose", json={"goal": "", "context": ""})
    assert resp.status_code in (400, 422)


def test_fix_propose_returns_invalid_when_bad_diff() -> None:
    bad = json.dumps({"explanation": "e", "diff": "nope"})
    gen = FixGenerator(llm=_StaticLLM(bad))
    app.dependency_overrides[get_fix_generator] = lambda: gen
    try:
        client = TestClient(app)
        resp = client.post("/fix/propose", json={"goal": "x", "context": ""})
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_valid"] is False
        assert body["validation_error"]
    finally:
        app.dependency_overrides.clear()