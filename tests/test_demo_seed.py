"""Loop 7 tests: the demo seed file, and restoring it (OFFLINE, fake embeddings)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from backend.indexing.ast_parser import parse_source
from backend.indexing.demo_seed import (
    DemoSeedError,
    export_seed,
    live_store_state,
    read_seed,
    restore_demo_seed,
    validate_seed_state,
)
from backend.indexing.stats import compute_stats
from backend.indexing.vector_store import VectorStore
from tests.test_index_safety import ALPHA, BETA, _store

MODEL = "models/gemini-embedding-001"
COMMIT = "611c616" + "0" * 33
TAGS = {
    "repo": "psf/requests", "ref": "main", "commit": COMMIT,
    "demo": "true", "indexed_at": "2026-10-18T10:00:00+00:00",
}


def _built(tmp: Path, dim: int = 6, model: str = MODEL) -> VectorStore:
    store = _store(tmp / "build", dim=dim, model=model)
    store.add_chunks(parse_source(ALPHA, Path("src/requests/a.py")), extra_metadata=TAGS)
    store.add_chunks(parse_source(BETA, Path("src/requests/b.py")), extra_metadata=TAGS)
    return store


def _seed(tmp: Path):
    store = _built(tmp)
    path = tmp / "seed" / "psf_requests.json.gz"
    manifest = export_seed(store, path, {"include": "src/"})
    return store, path, manifest


def _live_file(target: Path) -> Path:
    return target / "code_chunks" / "store.json"


def _load_seed(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))


def _write_seed(path: Path, state) -> None:
    path.write_bytes(gzip.compress(json.dumps(state).encode("utf-8")))


# --- building -------------------------------------------------------------


def test_export_writes_the_seed_and_a_readable_manifest(tmp_path: Path) -> None:
    store, path, manifest = _seed(tmp_path)
    assert path.is_file()
    assert manifest["repo"] == "psf/requests"
    assert manifest["commit"] == COMMIT
    assert (manifest["chunks"], manifest["files"], manifest["dimension"]) == (store.count(), 2, 6)
    assert manifest["model"] == MODEL and manifest["include"] == "src/"
    sidecar = json.loads((path.parent / "psf_requests.manifest.json").read_text(encoding="utf-8"))
    assert sidecar == manifest


def test_the_same_index_gives_the_same_bytes(tmp_path: Path) -> None:
    store = _built(tmp_path)
    one = tmp_path / "one.json.gz"
    two = tmp_path / "two.json.gz"
    export_seed(store, one, {"built_at": "fixed"})
    export_seed(store, two, {"built_at": "fixed"})
    assert one.read_bytes() == two.read_bytes()


def test_vectors_survive_the_file_exactly(tmp_path: Path) -> None:
    store, path, _ = _seed(tmp_path)
    original = store._vectors
    restored = np.asarray(_load_seed(path)["vectors"], dtype=np.float32)
    assert np.array_equal(original, restored)


def test_export_refuses_an_index_with_nothing_in_it(tmp_path: Path) -> None:
    with pytest.raises(DemoSeedError, match="nothing saved"):
        export_seed(_store(tmp_path / "empty"), tmp_path / "x.json.gz")


def test_no_temporary_files_are_left_after_export(tmp_path: Path) -> None:
    _, path, _ = _seed(tmp_path)
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "psf_requests.json.gz", "psf_requests.manifest.json",
    ]


# --- restoring into an empty place ---------------------------------------------


@pytest.mark.parametrize("prepare", ["missing", "zero_bytes", "empty_store"])
def test_restore_fills_a_missing_or_empty_store(tmp_path: Path, prepare: str) -> None:
    store, seed, _ = _seed(tmp_path)
    target = tmp_path / "live"
    state_file = _live_file(target)
    if prepare == "zero_bytes":
        state_file.parent.mkdir(parents=True)
        state_file.write_bytes(b"")
    elif prepare == "empty_store":
        state_file.parent.mkdir(parents=True)
        state_file.write_text(json.dumps({"ids": [], "documents": [], "metadatas": [], "vectors": []}))
    status = restore_demo_seed(seed, target)
    assert status.restored is True and status.seed_found and status.configured
    assert (status.repo, status.commit, status.chunks) == ("psf/requests", COMMIT, store.count())
    assert (status.model, status.dimension) == (MODEL, 6)
    assert [p.name for p in state_file.parent.iterdir()] == ["store.json"]


def test_a_restored_demo_is_searchable_and_marked_as_demo(tmp_path: Path) -> None:
    _, seed, _ = _seed(tmp_path)
    target = tmp_path / "live"
    restore_demo_seed(seed, target)
    live = _store(target, dim=6, model=MODEL, name="code_chunks")
    hits = live.query("alpha", top_k=3, repo="psf/requests")
    assert hits and hits[0].metadata["qualified_name"] == "alpha_fn"
    assert live.embedding_info() == {"model": MODEL, "dimension": 6}
    repo = compute_stats(live.all_metadata()).repos[0]
    assert (repo.repo, repo.is_demo, repo.commit, repo.ref) == ("psf/requests", True, COMMIT, "main")


# --- never touching a store that has data ---------------------------------------


def test_a_live_store_with_data_is_left_exactly_as_it_is(tmp_path: Path) -> None:
    _, seed, _ = _seed(tmp_path)
    target = tmp_path / "live"
    other = _store(target, dim=6, model=MODEL, name="code_chunks")
    other.add_chunks(parse_source(ALPHA, Path("x.py")), extra_metadata={"repo": "someone/else"})
    before = _live_file(target).read_bytes()
    status = restore_demo_seed(seed, target)
    assert status.restored is False and "already has data" in status.message
    assert _live_file(target).read_bytes() == before


def test_restoring_twice_changes_nothing_the_second_time(tmp_path: Path) -> None:
    _, seed, _ = _seed(tmp_path)
    target = tmp_path / "live"
    assert restore_demo_seed(seed, target).restored is True
    first = _live_file(target).read_bytes()
    again = restore_demo_seed(seed, target)
    assert again.restored is False and "already has data" in again.message
    assert _live_file(target).read_bytes() == first


def test_an_unreadable_small_live_file_is_not_overwritten(tmp_path: Path) -> None:
    _, seed, _ = _seed(tmp_path)
    target = tmp_path / "live"
    _live_file(target).parent.mkdir(parents=True)
    _live_file(target).write_text("{not json", encoding="utf-8")
    status = restore_demo_seed(seed, target)
    assert status.restored is False and "cannot be read" in status.message
    assert _live_file(target).read_text(encoding="utf-8") == "{not json"


def test_a_large_live_file_counts_as_data_without_being_parsed(tmp_path: Path) -> None:
    _, seed, _ = _seed(tmp_path)
    target = tmp_path / "live"
    _live_file(target).parent.mkdir(parents=True)
    _live_file(target).write_text("x" * 50_000, encoding="utf-8")
    assert live_store_state(_live_file(target)) == "has_data"
    assert restore_demo_seed(seed, target).restored is False
    assert _live_file(target).read_text(encoding="utf-8") == "x" * 50_000


# --- bad seeds are refused ---------------------------------------------------------


def _corrupt(state: dict, how: str):
    state = json.loads(json.dumps(state))
    if how == "format":
        state["format"] = 99
    elif how == "no_manifest":
        state.pop("manifest")
    elif how == "no_repo":
        state["manifest"]["repo"] = ""
    elif how == "missing_key":
        state.pop("documents")
    elif how == "length":
        state["ids"].pop()
    elif how == "zero_chunks":
        for key in ("ids", "documents", "metadatas", "vectors"):
            state[key] = []
    elif how == "ragged":
        state["vectors"][0] = state["vectors"][0][:-1]
    elif how == "repeated_ids":
        state["ids"][1] = state["ids"][0]
    elif how == "empty_vectors":
        state["vectors"] = [[] for _ in state["vectors"]]
    return state


@pytest.mark.parametrize("how", [
    "format", "no_manifest", "no_repo", "missing_key", "length",
    "zero_chunks", "ragged", "repeated_ids", "empty_vectors",
])
def test_a_damaged_seed_is_refused_and_the_live_store_is_untouched(tmp_path: Path, how: str) -> None:
    _, good, _ = _seed(tmp_path)
    bad = tmp_path / "bad.json.gz"
    _write_seed(bad, _corrupt(_load_seed(good), how))
    target = tmp_path / "live"
    status = restore_demo_seed(bad, target)
    assert status.restored is False and status.seed_found is True
    assert "refused" in status.message and "left unchanged" in status.message
    assert not _live_file(target).exists()


@pytest.mark.parametrize("content", [b"not gzip at all", gzip.compress(b"not json"), gzip.compress(b"[1,2]")])
def test_files_that_are_not_a_seed_are_refused(tmp_path: Path, content: bytes) -> None:
    bad = tmp_path / "bad.json.gz"
    bad.write_bytes(content)
    status = restore_demo_seed(bad, tmp_path / "live")
    assert status.restored is False and "refused" in status.message
    assert not (tmp_path / "live").exists()


def test_validate_and_read_report_clear_errors(tmp_path: Path) -> None:
    with pytest.raises(DemoSeedError, match="not found"):
        read_seed(tmp_path / "nope.json.gz")
    with pytest.raises(DemoSeedError, match="not a valid index"):
        validate_seed_state(["x"])


# --- switched off, missing, wrong model, failures -----------------------------------


def test_a_missing_seed_is_reported_not_fatal(tmp_path: Path) -> None:
    status = restore_demo_seed(tmp_path / "nope.json.gz", tmp_path / "live")
    assert (status.configured, status.seed_found, status.restored) == (True, False, False)
    assert "No demo seed file found" in status.message
    assert not (tmp_path / "live").exists()


def test_switching_the_restore_off_does_nothing(tmp_path: Path) -> None:
    _, seed, _ = _seed(tmp_path)
    status = restore_demo_seed(seed, tmp_path / "live", enabled=False)
    assert (status.configured, status.restored) == (False, False)
    assert not (tmp_path / "live").exists()


def test_a_model_mismatch_is_restored_but_warned_about(tmp_path: Path) -> None:
    _, seed, _ = _seed(tmp_path)
    status = restore_demo_seed(seed, tmp_path / "live", configured_model="gemini-embedding-002")
    assert status.restored is True
    assert "Searches will fail until they match" in status.warning
    assert MODEL in status.warning and "models/gemini-embedding-002" in status.warning


@pytest.mark.parametrize("configured", [None, "", "gemini-embedding-001", "models/gemini-embedding-001"])
def test_a_matching_or_unset_model_gives_no_warning(tmp_path: Path, configured) -> None:
    _, seed, _ = _seed(tmp_path)
    status = restore_demo_seed(seed, tmp_path / "live", configured_model=configured)
    assert status.restored is True and status.warning == ""


def test_a_failed_write_reports_it_and_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, seed, _ = _seed(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("backend.indexing.demo_seed.os.replace", boom)
    status = restore_demo_seed(seed, tmp_path / "live")
    assert status.restored is False and "could not be written" in status.message
    assert not _live_file(tmp_path / "live").exists()
