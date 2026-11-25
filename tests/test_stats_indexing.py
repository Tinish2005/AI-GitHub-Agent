"""Loop 6 tests: labels and imports saved at index time, and the /stats route (OFFLINE)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.cloning.cloner import FakeCloner
from backend.cloning.indexing_service import IndexingService, repo_label_from_url
from backend.github.models import RepoCoord
from backend.github.repo_indexer import GitHubRepoIndexer
from backend.indexing.ast_parser import parse_source
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.metadata import extract_metadata_for_chunks
from backend.indexing.vector_store import VectorStore
from backend.main import app, get_vector_store


class _FakeBackend:
    def encode(self, sentences: list, **kwargs) -> list:
        return [[float(len(s)) + i for i in range(EMBEDDING_DIMENSION)] for s in sentences]


SESSIONS = (
    "from .adapters import HTTPAdapter\n"
    "import os\n"
    "from requests.models import Request\n"
    "\n"
    "def send():\n"
    "    return os.sep\n"
)
ADAPTERS = "import socket\n\ndef connect():\n    return socket\n"


def _store(tmp_path: Path, name: str = "s") -> VectorStore:
    return VectorStore(
        persist_directory=tmp_path / "vdb",
        collection_name=f"{name}_{tmp_path.name}",
        embedding_service=EmbeddingService(backend=_FakeBackend()),
    )


# --- imports for files that are not on disk --------------------------------


def test_imports_are_read_from_text_when_the_file_is_not_on_disk() -> None:
    path = Path("src/requests/sessions.py")
    chunks = parse_source(SESSIONS, path)
    without = extract_metadata_for_chunks(chunks)
    assert without[0].imports == ()  # the old behaviour: nothing found on disk
    with_text = extract_metadata_for_chunks(chunks, file_sources={path: SESSIONS})
    assert with_text[0].imports == ("adapters", "os", "requests.models")


def test_given_text_wins_over_a_local_file_with_the_same_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A repo file named like one of OUR files must not borrow our imports.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("import secret_local_module\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    path = Path("src/mod.py")
    chunks = parse_source("import json\n\ndef f():\n    return 1\n", path)
    metas = extract_metadata_for_chunks(
        chunks, file_sources={path: "import json\n\ndef f():\n    return 1\n"},
    )
    assert metas[0].imports == ("json",)


def test_files_on_disk_still_work_without_file_sources(tmp_path: Path) -> None:
    f = tmp_path / "pkg" / "x.py"
    f.parent.mkdir()
    f.write_text("import os\n\ndef g():\n    return os.sep\n", encoding="utf-8")
    chunks = parse_source(f.read_text(encoding="utf-8"), f)
    assert extract_metadata_for_chunks(chunks, project_root=tmp_path)[0].imports == ("os",)


def test_broken_source_text_gives_no_imports_instead_of_an_error() -> None:
    path = Path("bad.py")
    chunks = parse_source("def f():\n    return 1\n", path)
    metas = extract_metadata_for_chunks(chunks, file_sources={path: "def (:\n"})
    assert metas[0].imports == ()


# --- the vector store ------------------------------------------------------


def test_extra_metadata_is_saved_with_every_chunk_and_survives_reload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_chunks(parse_source(ADAPTERS, Path("a.py")), extra_metadata={"repo": "o/r", "ref": "main"})
    reloaded = _store(tmp_path)
    metas = reloaded.all_metadata()
    assert metas and all(m["repo"] == "o/r" and m["ref"] == "main" for m in metas)


def test_add_chunks_without_extra_metadata_is_unchanged(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_chunks(parse_source(ADAPTERS, Path("a.py")))
    assert all("repo" not in m for m in store.all_metadata())


def test_all_metadata_is_empty_for_a_missing_store_and_returns_copies(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.all_metadata() == []
    store.add_chunks(parse_source(ADAPTERS, Path("a.py")))
    store.all_metadata()[0]["qualified_name"] = "changed"
    assert store.all_metadata()[0]["qualified_name"] != "changed"


def test_reindexing_with_the_same_label_updates_chunks_in_place(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunks = parse_source(ADAPTERS, Path("a.py"))
    store.add_chunks(chunks, extra_metadata={"repo": "o/r", "indexed_at": "2026-10-01T00:00:00+00:00"})
    store.add_chunks(chunks, extra_metadata={"repo": "o/r", "indexed_at": "2026-10-02T00:00:00+00:00"})
    metas = store.all_metadata()
    assert len(metas) == len(chunks)
    assert all(m["indexed_at"] == "2026-10-02T00:00:00+00:00" for m in metas)


# --- the GitHub indexer ----------------------------------------------------


def _github_indexer(store: VectorStore, monkeypatch: pytest.MonkeyPatch) -> GitHubRepoIndexer:
    indexer = GitHubRepoIndexer(token=None, vector_store=store, max_files=10)
    files = {"src/requests/sessions.py": SESSIONS, "src/requests/adapters.py": ADAPTERS}
    monkeypatch.setattr(indexer, "_default_branch", lambda owner, repo: "main")
    monkeypatch.setattr(indexer, "_list_python_files", lambda owner, repo, ref: list(files))
    monkeypatch.setattr(indexer, "_fetch_file", lambda owner, repo, path, ref: files[path])
    return indexer


def test_github_indexer_labels_chunks_and_keeps_their_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    result = _github_indexer(store, monkeypatch).index(RepoCoord(owner="psf", repo="requests"))
    assert result.chunks_indexed == len(store.all_metadata())
    for meta in store.all_metadata():
        assert meta["repo"] == "psf/requests"
        assert meta["ref"] == "main"
        assert meta["indexed_at"].endswith("+00:00")
    sessions = [m for m in store.all_metadata() if m["file_path"].replace("\\", "/").endswith("sessions.py")]
    assert sessions and sessions[0]["imports"] == "adapters,os,requests.models"


def test_github_indexer_with_nothing_to_index_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    indexer = GitHubRepoIndexer(token=None, vector_store=store)
    monkeypatch.setattr(indexer, "_default_branch", lambda owner, repo: "main")
    monkeypatch.setattr(indexer, "_list_python_files", lambda owner, repo, ref: [])
    assert indexer.index(RepoCoord(owner="o", repo="r")).chunks_indexed == 0
    assert store.all_metadata() == []


# --- the clone indexer ---------------------------------------------------------


@pytest.mark.parametrize("url,label", [
    ("https://github.com/fake/proj.git", "fake/proj"),
    ("https://github.com/psf/requests", "psf/requests"),
    ("https://github.com/psf/requests/", "psf/requests"),
    ("git@github.com:psf/requests.git", "psf/requests"),
    ("proj", "proj"),
    ("", ""),
])
def test_repo_label_from_url(url: str, label: str) -> None:
    assert repo_label_from_url(url) == label


def test_clone_indexer_labels_chunks(tmp_path: Path) -> None:
    store = _store(tmp_path, "clone")
    service = IndexingService(cloner=FakeCloner(cache_dir=tmp_path / "cache"), vector_store=store)
    service.index_repo("https://github.com/fake/proj.git")
    metas = store.all_metadata()
    assert metas
    assert all(m["repo"] == "fake/proj" and m["indexed_at"] and m["root"] for m in metas)


# --- GET /stats ----------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path) -> Iterator:
    store = _store(tmp_path, "route")
    app.dependency_overrides[get_vector_store] = lambda: store
    yield store, TestClient(app)
    app.dependency_overrides.clear()


def test_stats_route_on_an_empty_store_returns_the_empty_result(client) -> None:
    _, c = client
    resp = c.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    limits = body.pop("limits")  # the limits section has its own tests (test_index_jobs.py)
    assert set(limits) == {"public_indexing", "ai", "embeddings"}
    assert body == {
        "total_repos": 0, "total_files": 0, "total_chunks": 0, "repos": [],
        "embedding_model": None, "embedding_dimension": None, "demo_seed": None,
    }


def test_stats_route_returns_real_numbers_for_an_indexed_repo(
    client, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, c = client
    _github_indexer(store, monkeypatch).index(RepoCoord(owner="psf", repo="requests"))
    body = c.get("/stats").json()
    assert (body["total_repos"], body["total_files"]) == (1, 2)
    repo = body["repos"][0]
    assert repo["repo"] == "psf/requests" and repo["tracked"] is True
    assert repo["ref"] == "main" and repo["is_demo"] is False
    assert repo["last_synced"].endswith("+00:00")
    assert repo["chunks"] == body["total_chunks"] == len(store.all_metadata())
    assert {f["file_path"] for f in repo["chunks_per_file"]} == {
        "src/requests/sessions.py", "src/requests/adapters.py",
    }
    assert {"source": "src/requests/sessions.py", "target": "src/requests/adapters.py"} in repo["import_edges"]
    assert {"name": "os", "files": 1} in repo["top_external_imports"]


def test_stats_route_reports_old_unlabelled_chunks_as_untracked(client) -> None:
    store, c = client
    store.add_chunks(parse_source(ADAPTERS, Path("old.py")))
    body = c.get("/stats").json()
    assert body["total_repos"] == 0
    assert body["repos"][0]["repo"] == "(indexed before tracking)"
    assert body["repos"][0]["tracked"] is False


def test_stats_route_filters_by_repo_and_limits_rows(client, monkeypatch: pytest.MonkeyPatch) -> None:
    store, c = client
    _github_indexer(store, monkeypatch).index(RepoCoord(owner="psf", repo="requests"))
    assert c.get("/stats", params={"repo": "nope/none"}).json()["repos"] == []
    only = c.get("/stats", params={"repo": "PSF/Requests", "top": 1}).json()
    assert len(only["repos"][0]["chunks_per_file"]) == 1


@pytest.mark.parametrize("top", [0, 201, "x"])
def test_stats_route_rejects_a_bad_top(client, top) -> None:
    _, c = client
    assert c.get("/stats", params={"top": top}).status_code == 422


def test_stats_route_gives_a_clear_error_when_the_store_cannot_be_read(tmp_path: Path) -> None:
    class _Broken:
        def all_metadata(self):
            raise ValueError("corrupt")

    app.dependency_overrides[get_vector_store] = lambda: _Broken()
    try:
        resp = TestClient(app).get("/stats")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 500
    assert resp.json()["detail"] == "The index could not be read."


# --- the helper script ------------------------------------------------------------


def test_show_stats_script_renders_a_readable_report(client, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    from backend.indexing.stats import compute_stats

    script = Path(__file__).resolve().parent.parent / "scripts" / "show_stats.py"
    spec = importlib.util.spec_from_file_location("show_stats", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    store, _ = client
    assert "Nothing is indexed" in module.render(compute_stats(store.all_metadata()))
    _github_indexer(store, monkeypatch).index(RepoCoord(owner="psf", repo="requests"))
    text = module.render(compute_stats(store.all_metadata()))
    assert "== psf/requests" in text
    assert "src/requests/sessions.py -> src/requests/adapters.py" in text
    info = store.embedding_info()
    stats = compute_stats(store.all_metadata())
    stats.embedding_model, stats.embedding_dimension = info["model"], info["dimension"]
    assert f"Embeddings: {info['dimension']} numbers per chunk" in module.render(stats)
