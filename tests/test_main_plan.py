"""Integration tests for the /plan endpoint."""

from __future__ import annotations

from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.main import app, get_llm_client
from backend.rag.llm import EchoLLMClient


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_llm_client] = lambda: EchoLLMClient()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_plan_endpoint_default_strategy(client: TestClient) -> None:
    resp = client.post("/plan", json={"goal": "Fix the login bug"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["goal"] == "Fix the login bug"
    assert body["strategy"] == "rule_based"
    assert len(body["steps"]) > 0
    assert body["steps"][-1]["kind"] == "draft_pr"


def test_plan_endpoint_llm_strategy_falls_back(client: TestClient) -> None:
    """EchoLLMClient never returns valid JSON, so LLMPlanner should fall back."""
    resp = client.post("/plan", json={"goal": "Fix bug in auth", "strategy": "llm"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "rule_based"


def test_plan_endpoint_rejects_empty_goal(client: TestClient) -> None:
    resp = client.post("/plan", json={"goal": ""})
    assert resp.status_code in (400, 422)


def test_plan_endpoint_rejects_unknown_strategy(client: TestClient) -> None:
    resp = client.post(
        "/plan",
        json={"goal": "do something", "strategy": "psychic"},
    )
    assert resp.status_code == 400