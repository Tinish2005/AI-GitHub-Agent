"""Loop 8 tests: real AI calls are counted on the real routes (OFFLINE, a stub stands in for Gemini)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import backend.main as main_module
from backend.config import Settings, get_settings
from backend.indexing.ast_parser import parse_source
from backend.limits import LimitGuard
from backend.main import app, get_vector_store
from tests.test_index_safety import ALPHA, _store

KEY = "admin-key-for-tests"


class StubGemini:
    """Stands in for GeminiLLMClient: counts calls, can be told to fail."""

    calls = 0
    models: list = []
    error: Exception | None = None

    def __init__(self, api_key: str, *, model: str = "default", **kwargs) -> None:
        StubGemini.models.append(model)
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        StubGemini.calls += 1
        if StubGemini.error:
            raise StubGemini.error
        return "a stub answer"


class ResourceExhausted(Exception):
    pass


def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kw) -> TestClient:
    StubGemini.calls, StubGemini.models, StubGemini.error = 0, [], None
    monkeypatch.setattr(main_module, "GeminiLLMClient", StubGemini)
    cfg = Settings(_env_file=None, admin_api_key=SecretStr(KEY), trusted_proxy_hops=1, **kw)
    store = _store(tmp_path / "vdb")
    store.add_chunks(parse_source(ALPHA, Path("a.py")), extra_metadata={"repo": "o/r"})
    app.state.guard = LimitGuard(cfg)
    app.dependency_overrides[get_settings] = lambda: cfg
    app.dependency_overrides[get_vector_store] = lambda: store
    return TestClient(app)


@pytest.fixture(autouse=True)
def restore() -> Iterator:
    saved = app.state.guard
    yield
    app.state.guard = saved
    app.dependency_overrides.clear()


def ask(client: TestClient, ip: str = "5.5.5.5", **headers):
    return client.post("/qa", json={"question": "alpha"}, headers={"X-Forwarded-For": ip, **headers})


def gemini(**kw) -> dict:
    return {"gemini_api_key": SecretStr("k"), **kw}


def test_calls_are_counted_and_the_daily_limit_gives_a_clear_429(tmp_path, monkeypatch) -> None:
    client = setup(tmp_path, monkeypatch, **gemini(ai_calls_per_day=3, ai_calls_per_minute=10, ai_calls_per_ip_per_hour=10))
    for _ in range(3):
        assert ask(client).status_code == 200
    resp = ask(client)
    assert resp.status_code == 429 and "daily AI limit" in resp.json()["detail"]
    assert int(resp.headers["retry-after"]) > 0
    assert StubGemini.calls == 3, "the refused call must never reach the provider"


def test_the_stats_page_shows_how_many_calls_are_left(tmp_path, monkeypatch) -> None:
    client = setup(tmp_path, monkeypatch, **gemini(ai_calls_per_day=5))
    ask(client); ask(client)
    assert client.get("/stats").json()["limits"]["ai"]["calls_left_today"] == 3


def test_the_admin_key_is_not_counted_or_limited(tmp_path, monkeypatch) -> None:
    client = setup(tmp_path, monkeypatch, **gemini(ai_calls_per_day=1))
    for _ in range(4):
        assert ask(client, **{"X-Admin-Key": KEY}).status_code == 200
    assert app.state.guard.ai_day.remaining() == 1


def test_without_a_gemini_or_openai_key_nothing_is_counted(tmp_path, monkeypatch) -> None:
    client = setup(tmp_path, monkeypatch, ai_calls_per_day=1)
    for _ in range(4):
        assert ask(client).status_code == 200
    assert app.state.guard.ai_day.remaining() == 1 and StubGemini.calls == 0


def test_one_address_is_limited_per_hour_and_a_forged_header_does_not_help(tmp_path, monkeypatch) -> None:
    client = setup(tmp_path, monkeypatch, **gemini(ai_calls_per_ip_per_hour=2, ai_calls_per_minute=50, ai_calls_per_day=50))
    assert ask(client, "1.1.1.1, 7.7.7.7").status_code == 200
    assert ask(client, "2.2.2.2, 7.7.7.7").status_code == 200       # a different forged start, same real address
    resp = ask(client, "3.3.3.3, 7.7.7.7")
    assert resp.status_code == 429 and "this hour" in resp.json()["detail"]
    assert ask(client, "8.8.8.8").status_code == 200                 # a different real address


def test_too_many_calls_in_a_minute_say_the_ai_is_busy(tmp_path, monkeypatch) -> None:
    client = setup(tmp_path, monkeypatch, **gemini(ai_calls_per_minute=1, ai_calls_per_ip_per_hour=50, ai_calls_per_day=50))
    assert ask(client, "1.1.1.1").status_code == 200
    resp = ask(client, "2.2.2.2")
    assert resp.status_code == 429 and "busy" in resp.json()["detail"]


def test_a_provider_quota_error_is_a_429_not_a_crash(tmp_path, monkeypatch) -> None:
    client = setup(tmp_path, monkeypatch, **gemini())
    StubGemini.error = ResourceExhausted("429 You exceeded your current quota")
    resp = ask(client)
    assert resp.status_code == 429 and "AI provider" in resp.json()["detail"]


def test_the_ai_model_comes_from_the_settings(tmp_path, monkeypatch) -> None:
    client = setup(tmp_path, monkeypatch, **gemini(gemini_llm_model="my-special-model"))
    ask(client)
    assert StubGemini.models[-1] == "my-special-model"


def test_the_default_ai_model_is_unchanged() -> None:
    assert Settings(_env_file=None).gemini_llm_model == "gemini-2.5-flash"


def test_a_run_that_hits_the_daily_limit_part_way_reports_it_in_the_step(tmp_path, monkeypatch) -> None:
    client = setup(tmp_path, monkeypatch, **gemini(ai_calls_per_day=1, ai_calls_per_minute=10, ai_calls_per_ip_per_hour=10))
    body = client.post("/execute", json={"goal": "fix the alpha bug", "owner": "o", "repo": "r"},
                       headers={"X-Forwarded-For": "5.5.5.5"}).json()
    steps = {s["kind"]: s for s in body["steps"]}
    assert steps["analyze"]["status"] == "completed"           # the one allowed call
    assert steps["generate"]["status"] == "failed" and "daily AI limit" in steps["generate"]["error"]
    assert body["aborted"] is True and StubGemini.calls == 1
