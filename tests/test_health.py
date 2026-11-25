"""/health answers from memory only, for GET and HEAD."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import backend.main as main
from backend.config import get_settings
from backend.indexing.demo_seed import DemoSeedStatus
from backend.indexing.vector_store import VectorStore


@pytest.fixture
def client(monkeypatch):
    """The real app, with every route to the index, the AI and GitHub rigged to explode."""

    def boom(*args, **kwargs):
        raise AssertionError("/health must not touch the index, the AI or GitHub")

    monkeypatch.setattr(VectorStore, "_ensure_loaded", boom)
    monkeypatch.setattr(main, "VectorStore", boom)
    monkeypatch.setattr(main, "build_embedding_service", boom)
    monkeypatch.setattr(main, "GeminiLLMClient", boom)
    monkeypatch.setattr(main, "OpenAILLMClient", boom)
    monkeypatch.setattr(main, "GitHubClient", boom)
    monkeypatch.setattr(main, "GitHubSnapshotProvider", boom)
    return TestClient(main.app)


@pytest.mark.parametrize("path", ["/health", "/"])
def test_get_never_touches_index_ai_or_github(client, path):
    response = client.get(path)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == get_settings().app_version  # one source of truth
    assert body["app_name"] and body["environment"]
    assert isinstance(body["uptime_seconds"], (int, float)) and body["uptime_seconds"] >= 0
    assert "demo_restore" in body and "demo_message" in body and "restore_seconds" in body


@pytest.mark.parametrize("path", ["/health", "/"])
def test_head_works_for_uptime_monitors(client, path):
    response = client.head(path)
    assert response.status_code == 200
    assert response.content == b""


def test_fields_are_documented_in_openapi(client):
    schema = client.get("/openapi.json").json()
    props = schema["components"]["schemas"]["HealthResponse"]["properties"]
    for name in ("status", "uptime_seconds", "demo_restore", "demo_message", "restore_seconds"):
        assert props[name].get("description"), name
    assert "get" in schema["paths"]["/health"]


def test_reports_uptime_and_restore_status(client, monkeypatch):
    state = main.app.state
    monkeypatch.setattr(state, "ready_at", time.monotonic() - 125, raising=False)
    monkeypatch.setattr(state, "restore_seconds", 0.42, raising=False)
    monkeypatch.setattr(state, "demo_restore_failed", False, raising=False)
    monkeypatch.setattr(
        state, "demo_status",
        DemoSeedStatus(configured=True, seed_found=True, restored=True, message="Demo restored: psf/requests (9 chunks)."),
        raising=False,
    )
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert 125 <= body["uptime_seconds"] < 135
    assert body["demo_restore"] == "restored"
    assert body["demo_message"].startswith("Demo restored")
    assert body["restore_seconds"] == 0.42


def test_a_failed_restore_still_answers_ok(client, monkeypatch):
    state = main.app.state
    monkeypatch.setattr(state, "demo_restore_failed", True, raising=False)
    monkeypatch.setattr(
        state, "demo_status",
        DemoSeedStatus(configured=True, message="The demo could not be restored (OSError)."),
        raising=False,
    )
    body = client.get("/health").json()
    assert body["status"] == "ok"  # ok means "ready"; the detail is in demo_restore
    assert body["demo_restore"] == "failed"


@pytest.mark.parametrize(
    ("status", "failed", "expected"),
    [
        (None, False, "unknown"),
        (None, True, "failed"),
        (DemoSeedStatus(configured=False), False, "disabled"),
        (DemoSeedStatus(configured=True, seed_found=False), False, "seed_missing"),
        (DemoSeedStatus(configured=True, seed_found=True, restored=False), False, "unchanged"),
        (DemoSeedStatus(configured=True, seed_found=True, restored=True), False, "restored"),
    ],
)
def test_demo_restore_label(status, failed, expected):
    assert main.demo_restore_label(status, failed) == expected
