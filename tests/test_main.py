"""Integration tests for `backend.main` using FastAPI's TestClient."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def test_root_endpoint_returns_ok() -> None:
    """GET / should return a 200 with the expected health payload."""
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"]
    assert body["version"]
    assert body["environment"]


def test_health_endpoint_returns_ok() -> None:
    """GET /health should return a 200 with the expected health payload."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"]
    assert body["version"]
    assert body["environment"]


def test_openapi_schema_is_generated() -> None:
    """FastAPI should auto-generate an OpenAPI schema at /openapi.json."""
    client = TestClient(app)
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/health" in schema["paths"]