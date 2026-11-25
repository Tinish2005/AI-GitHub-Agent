"""CORS_ORIGINS and CORS_ORIGIN_REGEX behave as documented."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.cors import build_cors_config, parse_origins

VERCEL = "https://my-app.vercel.app"


def _client(origins: str = "", regex: str = "") -> TestClient:
    app = FastAPI()
    app.add_middleware(CORSMiddleware, **build_cors_config(origins, regex).middleware_kwargs())

    @app.get("/ping")
    def ping() -> dict:
        return {"ok": True}

    return TestClient(app)


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/ping",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-admin-key",
        },
    )


def test_default_allows_everyone_with_credentials_off():
    config = build_cors_config("", "")
    assert config.wildcard and config.allow_origins == ["*"]
    assert config.allow_credentials is False

    response = _preflight(_client(), "https://anything.example")
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


def test_star_wins_even_next_to_other_entries():
    config = build_cors_config(f"{VERCEL}, *")
    assert config.wildcard and config.allow_credentials is False


def test_listed_origin_is_allowed_and_others_are_not():
    client = _client(VERCEL)
    allowed = _preflight(client, VERCEL)
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == VERCEL
    assert allowed.headers["access-control-allow-credentials"] == "true"
    assert "x-admin-key" in allowed.headers["access-control-allow-headers"].lower()

    blocked = _preflight(client, "https://evil.example")
    assert "access-control-allow-origin" not in blocked.headers


def test_normal_request_gets_the_header_too():
    response = _client(VERCEL).get("/ping", headers={"Origin": VERCEL})
    assert response.headers["access-control-allow-origin"] == VERCEL


def test_entries_are_cleaned_up():
    assert parse_origins(' https://A.vercel.app/ , "https://b.com" ,, https://b.com ') == [
        "https://a.vercel.app",
        "https://b.com",
    ]
    # a trailing slash is the classic "why is it blocked" mistake
    response = _preflight(_client(VERCEL + "/"), VERCEL)
    assert response.headers["access-control-allow-origin"] == VERCEL


@pytest.mark.parametrize("bad", ["my-app.vercel.app", "https://x.com/app", "ftp://x.com", "https://"])
def test_bad_origin_fails_loudly(bad):
    with pytest.raises(ValueError, match="CORS_ORIGINS"):
        build_cors_config(bad)


def test_regex_allows_preview_urls():
    client = _client(VERCEL, r"^https://my-app(-[a-z0-9-]+)?\.vercel\.app$")
    preview = "https://my-app-git-feature-me.vercel.app"
    assert _preflight(client, preview).headers["access-control-allow-origin"] == preview
    assert _preflight(client, VERCEL).headers["access-control-allow-origin"] == VERCEL
    assert "access-control-allow-origin" not in _preflight(client, "https://other.vercel.app").headers


def test_regex_alone_is_not_allow_all():
    config = build_cors_config("", r"^https://x\.vercel\.app$")
    assert not config.wildcard and config.allow_origins == []


def test_bad_regex_fails_loudly():
    with pytest.raises(ValueError, match="CORS_ORIGIN_REGEX"):
        build_cors_config("", "([")


def test_settings_read_the_environment(monkeypatch):
    assert Settings(_env_file=None).cors_origins == "*"
    monkeypatch.setenv("CORS_ORIGINS", VERCEL)
    monkeypatch.setenv("CORS_ORIGIN_REGEX", r"^https://p\.vercel\.app$")
    settings = Settings(_env_file=None)
    assert settings.cors_origins == VERCEL
    assert settings.cors_origin_regex == r"^https://p\.vercel\.app$"

@pytest.mark.parametrize("origins", ["", VERCEL])
def test_retry_after_header_is_readable_by_the_page(origins):
    """A 429 carries Retry-After; the browser only lets the frontend read it if it is exposed."""
    app = FastAPI()
    app.add_middleware(CORSMiddleware, **build_cors_config(origins).middleware_kwargs())

    @app.get("/limited")
    def limited():
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": "busy"}, status_code=429, headers={"Retry-After": "30"})

    response = TestClient(app).get("/limited", headers={"Origin": VERCEL})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"
    assert "retry-after" in response.headers["access-control-expose-headers"].lower()
