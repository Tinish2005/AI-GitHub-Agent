"""Visitor indexing can be limited to one folder of a repository."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.config import Settings
from backend.indexing.jobs import IndexJobManager, normalize_folder
from backend.limits import LimitError, LimitGuard


class _Store:
    def __init__(self, repos=None):
        self.repos = repos or {}

    def repo_summaries(self):
        return self.repos

    def count(self):
        return sum(r["chunks"] for r in self.repos.values())

    def remove_repo(self, name):
        self.repos.pop(name, None)


class _Indexer:
    """Records how it was called. Accepts the new folder argument."""

    def __init__(self, chunks=10):
        self.calls = []
        self.chunks = chunks

    def index(self, coord, max_chunks, max_file_bytes, progress, before_add, include_prefixes=None):
        self.calls.append({"coord": coord, "include_prefixes": include_prefixes})
        return SimpleNamespace(files_indexed=2 if self.chunks else 0, chunks_indexed=self.chunks, files_skipped=0, truncated=False)


class _OldIndexer:
    """Like the indexers in older tests: it has no folder argument at all."""

    def __init__(self):
        self.calls = 0

    def index(self, coord, max_chunks, max_file_bytes, progress, before_add):
        self.calls += 1
        return SimpleNamespace(files_indexed=1, chunks_indexed=5, files_skipped=0, truncated=False)


def _manager(indexer, repos=None):
    settings = Settings(_env_file=None, public_indexing_enabled=True)
    guard = LimitGuard(settings)
    store = _Store(repos)
    manager = IndexJobManager(
        guard=guard, settings=settings, store_factory=lambda: store, indexer_factory=lambda s, n: indexer,
        repo_checker=lambda owner, repo: {"full_name": f"{owner}/{repo}"}, run_in_thread=False,
    )
    return manager, guard


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("", ""), ("   ", ""), ("/", ""), ("src/pkg", "src/pkg/"), ("src/pkg/", "src/pkg/"), ("/src/pkg/", "src/pkg/"),
        ("./src/pkg", "src/pkg/"), ("src\\pkg", "src/pkg/"), ("torch/nn", "torch/nn/"),
    ],
)
def test_folder_is_cleaned_up(typed, expected):
    assert normalize_folder(typed) == expected


@pytest.mark.parametrize("bad", ["../secrets", "src/../..", "a//b", "src/pkg;rm", "x" * 201, "src/./pkg", "a b", "src/\u00e9"])
def test_unsafe_folders_are_refused_with_400(bad):
    with pytest.raises(LimitError) as err:
        normalize_folder(bad)
    assert err.value.status == 400


def test_the_folder_reaches_the_indexer_as_a_prefix():
    indexer = _Indexer()
    manager, _ = _manager(indexer)
    view = manager.submit("pytorch", "pytorch", ip="1.1.1.1", path="torch/nn")
    assert view["status"] == "done"
    assert indexer.calls[0]["include_prefixes"] == ("torch/nn/",)


def test_without_a_folder_old_indexers_still_work():
    indexer = _OldIndexer()
    manager, _ = _manager(indexer)
    assert manager.submit("psf", "flask", ip="1.1.1.1")["status"] == "done"
    assert indexer.calls == 1  # the folder argument was not passed


def test_bad_folder_refused_before_anything_is_charged_or_locked():
    indexer = _Indexer()
    manager, guard = _manager(indexer)
    with pytest.raises(LimitError) as err:
        manager.submit("a", "b", ip="1.1.1.1", path="../x")
    assert err.value.status == 400
    assert indexer.calls == []
    assert guard.index_lock.acquire(blocking=False)  # the lock is free
    guard.index_lock.release()


def test_an_indexed_repo_is_reported_unless_a_folder_is_named():
    repos = {"pytorch/pytorch": {"demo": False, "chunks": 120, "indexed_at": "2026-11-01"}}
    indexer = _Indexer()
    manager, _ = _manager(indexer, repos)
    assert manager.submit("pytorch", "pytorch", ip="1.1.1.1")["status"] == "already_indexed"
    assert indexer.calls == []
    view = manager.submit("pytorch", "pytorch", ip="1.1.1.1", path="torch/nn")  # adds that part
    assert view["status"] == "done" and len(indexer.calls) == 1


def test_the_demo_cannot_be_indexed_again_even_with_a_folder():
    repos = {"psf/requests": {"demo": True, "chunks": 283, "indexed_at": "2026-10-19"}}
    manager, _ = _manager(_Indexer(), repos)
    with pytest.raises(LimitError) as err:
        manager.submit("psf", "requests", ip="1.1.1.1", path="src/requests")
    assert err.value.status == 409


def test_an_empty_folder_says_so():
    manager, _ = _manager(_Indexer(chunks=0))
    view = manager.submit("pytorch", "pytorch", ip="1.1.1.1", path="no/such/folder")
    assert view["status"] == "failed"
    assert "folder 'no/such/folder'" in view["error"]


def test_empty_repo_without_a_folder_keeps_the_old_message():
    manager, _ = _manager(_Indexer(chunks=0))
    view = manager.submit("a", "b", ip="1.1.1.1")
    assert view["error"] == "No Python code was found to index in the first files of this repository."


def test_the_request_model_accepts_an_optional_path():
    from pydantic import ValidationError

    from backend.main import PublicIndexRequest

    assert PublicIndexRequest(owner="a", repo="b").path == ""
    assert PublicIndexRequest(owner="a", repo="b", path="src/x").path == "src/x"
    with pytest.raises(ValidationError):
        PublicIndexRequest(owner="a", repo="b", path="x" * 201)


def test_the_route_passes_the_folder_only_when_given(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PUBLIC_INDEXING_ENABLED", "true")
    monkeypatch.setenv("RESTORE_DEMO_ON_STARTUP", "false")
    from backend.config import get_settings

    if hasattr(get_settings, "cache_clear"):
        get_settings.cache_clear()
    from backend.main import create_app

    app = create_app()
    seen = []

    class FakeJobs:
        def submit(self, owner, repo, **kwargs):
            seen.append(kwargs)
            return {"status": "running", "job_id": "j1", "repo": f"{owner}/{repo}"}

    app.state.jobs = FakeJobs()
    client = TestClient(app)
    assert client.post("/index/public", json={"owner": "a", "repo": "b"}).status_code == 202
    assert client.post("/index/public", json={"owner": "a", "repo": "b", "path": "src/x"}).status_code == 202
    assert "path" not in seen[0] and seen[1]["path"] == "src/x"
    if hasattr(get_settings, "cache_clear"):
        get_settings.cache_clear()
