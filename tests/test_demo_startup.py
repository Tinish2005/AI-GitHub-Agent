"""Loop 7 tests: the demo is restored when the server starts (OFFLINE)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.config import Settings, get_settings
from backend.indexing.ast_parser import parse_source
from backend.indexing.demo_seed import export_seed
from tests.test_demo_seed import MODEL, TAGS, _built
from tests.test_index_safety import ALPHA, _store


def _settings(tmp: Path, seed: Path, **extra) -> Settings:
    return Settings(
        _env_file=None,
        vector_db_path=tmp / "vdb",
        repo_cache_path=tmp / "cache",
        demo_seed_path=seed,
        **extra,
    )


def _app(monkeypatch: pytest.MonkeyPatch, settings: Settings):
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    app = main_module.create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    return app


@pytest.fixture
def seed(tmp_path: Path) -> Path:
    path = tmp_path / "seed" / "psf_requests.json.gz"
    export_seed(_built(tmp_path), path)
    return path


def test_a_fresh_start_restores_the_demo_and_stats_shows_it(
    tmp_path: Path, seed: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(monkeypatch, _settings(tmp_path, seed))
    with TestClient(app) as client:
        body = client.get("/stats").json()
    demo = body["demo_seed"]
    assert demo["restored"] is True and demo["seed_found"] is True
    assert demo["repo"] == "psf/requests" and demo["chunks"] == body["total_chunks"] > 0
    repo = body["repos"][0]
    assert (repo["repo"], repo["is_demo"], repo["tracked"]) == ("psf/requests", True, True)
    assert repo["commit"] == TAGS["commit"]
    assert (body["embedding_model"], body["embedding_dimension"]) == (MODEL, 6)


def test_the_restored_demo_answers_searches_for_its_repo(
    tmp_path: Path, seed: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path, seed)
    with TestClient(_app(monkeypatch, settings)):
        pass
    live = _store(settings.vector_db_path, dim=6, model=MODEL, name="code_chunks")
    hits = live.query("alpha", top_k=3, repo="psf/requests")
    assert hits and hits[0].metadata["demo"] == "true"


def test_a_second_start_leaves_the_restored_demo_alone(
    tmp_path: Path, seed: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path, seed)
    with TestClient(_app(monkeypatch, settings)):
        pass
    state_file = settings.vector_db_path / "code_chunks" / "store.json"
    before = state_file.read_bytes()
    with TestClient(_app(monkeypatch, settings)) as client:
        demo = client.get("/stats").json()["demo_seed"]
    assert demo["restored"] is False and "already has data" in demo["message"]
    assert state_file.read_bytes() == before


def test_existing_live_data_is_never_replaced_at_startup(
    tmp_path: Path, seed: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path, seed)
    other = _store(settings.vector_db_path, dim=6, model=MODEL, name="code_chunks")
    other.add_chunks(parse_source(ALPHA, Path("x.py")), extra_metadata={"repo": "someone/else"})
    state_file = settings.vector_db_path / "code_chunks" / "store.json"
    before = state_file.read_bytes()
    with TestClient(_app(monkeypatch, settings)) as client:
        body = client.get("/stats").json()
    assert body["demo_seed"]["restored"] is False
    assert [r["repo"] for r in body["repos"]] == ["someone/else"]
    assert state_file.read_bytes() == before


def test_a_missing_seed_file_does_not_stop_the_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(monkeypatch, _settings(tmp_path, tmp_path / "nope.json.gz"))
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        body = client.get("/stats").json()
    assert body["demo_seed"]["seed_found"] is False
    assert "No demo seed file found" in body["demo_seed"]["message"]
    assert body["total_chunks"] == 0


def test_the_restore_can_be_switched_off(
    tmp_path: Path, seed: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(monkeypatch, _settings(tmp_path, seed, restore_demo_on_startup=False))
    with TestClient(app) as client:
        body = client.get("/stats").json()
    assert body["demo_seed"]["configured"] is False
    assert body["total_chunks"] == 0


def test_a_crash_inside_the_restore_never_stops_the_server(
    tmp_path: Path, seed: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(main_module, "restore_demo_seed", boom)
    app = _app(monkeypatch, _settings(tmp_path, seed))
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        demo = client.get("/stats").json()["demo_seed"]
    assert demo["restored"] is False and "could not be restored" in demo["message"]


def test_a_model_setting_that_differs_from_the_demo_is_shown_as_a_warning(
    tmp_path: Path, seed: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(monkeypatch, _settings(tmp_path, seed, gemini_embedding_model="gemini-embedding-002"))
    with TestClient(app) as client:
        demo = client.get("/stats").json()["demo_seed"]
    assert demo["restored"] is True
    assert "Searches will fail until they match" in demo["warning"]


def test_without_a_startup_the_demo_section_is_empty(
    tmp_path: Path, seed: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(monkeypatch, _settings(tmp_path, seed))
    body = TestClient(app).get("/stats").json()
    assert body["demo_seed"] is None
