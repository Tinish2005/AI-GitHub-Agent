"""Loop 7 tests: indexing is closed unless the admin key is sent (OFFLINE)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import backend.main as main_module
from backend.cloning.cloner import FakeCloner
from backend.cloning.indexing_service import IndexingService
from backend.config import Settings, get_settings
from backend.main import ADMIN_DENIED_MESSAGE, _is_admin, app, get_indexing_service, get_mcp_server, get_vector_store
from backend.mcp.server import MCPServer
from backend.mcp.tools import Tool, ToolRegistry
from tests.test_index_safety import _store

KEY = "correct-horse-battery"
URL = {"url": "https://github.com/fake/proj.git"}
GITHUB = {"owner": "o", "repo": "r", "max_files": 3}


def _settings(key: str) -> Settings:
    return Settings(_env_file=None, admin_api_key=SecretStr(key))


def _tool(name: str) -> Tool:
    return Tool(
        name=name, description=name, input_schema={"type": "object", "properties": {}},
        execute=lambda params: f"{name} ran",
    )


@pytest.fixture
def env(tmp_path: Path) -> Iterator:
    store = _store(tmp_path / "vdb")
    indexer = IndexingService(cloner=FakeCloner(cache_dir=tmp_path / "cache"), vector_store=store)
    registry = ToolRegistry()
    registry.register(_tool("index_repo"))
    registry.register(_tool("search_code"))
    app.dependency_overrides[get_indexing_service] = lambda: indexer
    app.dependency_overrides[get_mcp_server] = lambda: MCPServer(registry)
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: _settings(KEY)
    yield SimpleNamespace(client=TestClient(app), store=store)
    app.dependency_overrides.clear()


def _use_key(key: str) -> None:
    app.dependency_overrides[get_settings] = lambda: _settings(key)


# --- the two HTTP doors -----------------------------------------------------


@pytest.mark.parametrize("route,body", [("/index", URL), ("/index/github", GITHUB)])
def test_indexing_without_the_key_is_refused(env, route: str, body: dict) -> None:
    resp = env.client.post(route, json=body)
    assert resp.status_code == 403
    assert resp.json()["detail"] == ADMIN_DENIED_MESSAGE


@pytest.mark.parametrize("route,body", [("/index", URL), ("/index/github", GITHUB)])
def test_indexing_with_a_wrong_key_is_refused(env, route: str, body: dict) -> None:
    for wrong in ["wrong", KEY.upper(), KEY + " ", ""]:
        resp = env.client.post(route, json=body, headers={"X-Admin-Key": wrong})
        assert resp.status_code == 403, wrong


@pytest.mark.parametrize("route,body", [("/index", URL), ("/index/github", GITHUB)])
def test_indexing_is_closed_when_no_key_is_configured(env, route: str, body: dict) -> None:
    _use_key("")
    for header in [{}, {"X-Admin-Key": ""}, {"X-Admin-Key": KEY}, {"X-Admin-Key": "anything"}]:
        assert env.client.post(route, json=body, headers=header).status_code == 403


def test_the_right_key_opens_the_clone_route(env) -> None:
    resp = env.client.post("/index", json=URL, headers={"X-Admin-Key": KEY})
    assert resp.status_code == 200 and resp.json()["chunks_indexed"] > 0


def test_the_right_key_opens_the_github_route(env, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Indexer:
        def __init__(self, token, vector_store, max_files):
            pass

        def index(self, coord):
            return SimpleNamespace(owner=coord.owner, repo=coord.repo, ref="main",
                                   files_indexed=1, chunks_indexed=2)

    monkeypatch.setattr(main_module, "GitHubRepoIndexer", _Indexer)
    resp = env.client.post("/index/github", json=GITHUB, headers={"X-Admin-Key": KEY})
    assert resp.status_code == 200 and resp.json()["chunks_indexed"] == 2


def test_a_refused_request_adds_nothing_to_the_index(env) -> None:
    env.client.post("/index", json=URL)
    assert env.store.count() == 0


# --- the MCP door -----------------------------------------------------------------


def _call(client: TestClient, tool: str, headers: dict | None = None) -> dict:
    return client.post(
        "/mcp", headers=headers or {},
        json={"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": tool, "arguments": {}}},
    ).json()


def test_the_mcp_index_tool_needs_the_key(env) -> None:
    body = _call(env.client, "index_repo")
    assert body["id"] == 7 and body["result"] is None
    assert body["error"]["message"] == ADMIN_DENIED_MESSAGE
    wrong = _call(env.client, "index_repo", {"X-Admin-Key": "wrong"})
    assert wrong["error"]["message"] == ADMIN_DENIED_MESSAGE


def test_the_mcp_index_tool_runs_with_the_key(env) -> None:
    body = _call(env.client, "index_repo", {"X-Admin-Key": KEY})
    assert body.get("error") is None and "index_repo ran" in str(body["result"])


def test_the_mcp_index_tool_is_closed_when_no_key_is_configured(env) -> None:
    _use_key("")
    assert _call(env.client, "index_repo", {"X-Admin-Key": ""})["error"] is not None


def test_other_mcp_tools_and_listing_need_no_key(env) -> None:
    assert "search_code ran" in str(_call(env.client, "search_code")["result"])
    listing = env.client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).json()
    assert {t["name"] for t in listing["result"]["tools"]} == {"index_repo", "search_code"}


# --- everything else stays open ----------------------------------------------------


def test_stats_and_the_root_page_stay_open(env) -> None:
    assert env.client.get("/stats").status_code == 200
    assert env.client.get("/").status_code == 200


@pytest.mark.parametrize("supplied,expected", [
    (KEY, True), ("nope", False), ("", False), (None, False), (KEY[:-1], False),
])
def test_key_comparison(supplied, expected) -> None:
    assert _is_admin(_settings(KEY), supplied) is expected


def test_no_configured_key_never_matches() -> None:
    assert _is_admin(_settings(""), "") is False
    assert _is_admin(_settings(""), "x") is False


def test_non_ascii_keys_work() -> None:
    assert _is_admin(_settings("schlüssel-ключ"), "schlüssel-ключ") is True
    assert _is_admin(_settings("schlüssel-ключ"), "schlussel") is False
