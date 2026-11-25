"""The index is cached in memory and re-read whenever the file changes."""

from __future__ import annotations

import json
import logging
import os

import pytest

from backend.indexing import vector_store as vs
from backend.indexing.vector_store import VectorStore, clear_index_cache


class _NoEmbeddings:
    """Stands in for the embedding service. These tests never embed anything."""

    active_model_name = "test-model"


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_index_cache()
    yield
    clear_index_cache()


@pytest.fixture
def loads(monkeypatch):
    """Counts how many times the index file is actually read and parsed."""
    calls = []
    real = vs._read_state_file

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(vs, "_read_state_file", counting)
    return calls


def _write_state(persist_dir, n, *, tag="a", model="models/test"):
    """Write store.json the way the indexer does: temp file, then os.replace."""
    path = persist_dir / vs.DEFAULT_COLLECTION / "store.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "ids": [f"id{i}" for i in range(n)],
        "documents": [f"doc {tag} {i}" for i in range(n)],
        "metadatas": [{"repo": "o/a" if i % 2 == 0 else "o/b", "name": f"f{i}"} for i in range(n)],
        "vectors": [[1.0, 0.0, float(i)] for i in range(n)],
        "embedding": {"model": model},
    }
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(temp, path)
    return path


def _store(persist_dir):
    return VectorStore(persist_dir, embedding_service=_NoEmbeddings())


def test_many_stores_share_one_read(tmp_path, loads):
    _write_state(tmp_path, 4)
    for _ in range(5):  # the API builds a new VectorStore per request
        assert _store(tmp_path).count() == 4
    assert len(loads) == 1


def test_unchanged_file_is_not_read_again(tmp_path, loads):
    _write_state(tmp_path, 4)
    store = _store(tmp_path)
    for _ in range(10):
        store.count()
        store.all_metadata()
        store.embedding_info()
    assert len(loads) == 1


def test_outside_write_is_picked_up_by_new_and_existing_stores(tmp_path, loads):
    _write_state(tmp_path, 2)
    old = _store(tmp_path)
    assert old.count() == 2

    _write_state(tmp_path, 5)  # another process or job finished indexing

    assert old.count() == 5  # the store that was already open sees it
    assert _store(tmp_path).count() == 5  # and so does a new one
    assert len(loads) == 2  # one read for each version of the file, no more


def test_rewrite_with_the_same_size_is_detected(tmp_path, loads):
    path = _write_state(tmp_path, 3, tag="a")
    store = _store(tmp_path)
    assert store.all_metadata()[0]["repo"] == "o/a"
    before = path.stat().st_size

    _write_state(tmp_path, 3, tag="b")  # same size, different content
    assert path.stat().st_size == before
    stamp = path.stat().st_mtime_ns + 1_000_000_000  # a coarse clock must not hide it
    os.utime(path, ns=(stamp, stamp))

    store.count()
    assert len(loads) == 2


def test_removing_a_repo_is_seen_by_other_stores(tmp_path):
    _write_state(tmp_path, 4)
    reader, writer = _store(tmp_path), _store(tmp_path)
    assert reader.count() == 4

    assert writer.remove_repo("o/a") == 2

    assert reader.count() == 2
    assert _store(tmp_path).count() == 2


def test_own_write_does_not_trigger_a_reload(tmp_path, loads):
    _write_state(tmp_path, 4)
    store = _store(tmp_path)
    store.remove_repo("o/a")
    store.count()
    store.count()
    assert len(loads) == 1  # the first read only


def test_callers_cannot_change_the_cached_copy(tmp_path):
    _write_state(tmp_path, 3)
    first = _store(tmp_path)
    first.all_metadata()[0]["repo"] = "tampered"
    assert _store(tmp_path).all_metadata()[0]["repo"] == "o/a"

    assert first._vectors.flags.writeable is False  # shared copy is read-only
    first._own_vectors()  # what add_chunks does before changing vectors
    first._vectors[0, 0] = 99.0
    other = _store(tmp_path)
    other.count()
    assert other._vectors[0, 0] == 1.0  # the cached copy is untouched


def test_unreadable_file_keeps_serving_the_last_good_copy(tmp_path, caplog):
    path = _write_state(tmp_path, 3)
    assert _store(tmp_path).count() == 3

    path.write_text('{"ids": ["half', encoding="utf-8")  # caught mid-write
    with caplog.at_level(logging.WARNING, logger=vs.logger.name):
        assert _store(tmp_path).count() == 3
    assert "previous copy" in caplog.text

    _write_state(tmp_path, 6)  # the writer finishes
    assert _store(tmp_path).count() == 6


def test_unreadable_file_with_no_earlier_copy_raises(tmp_path):
    path = tmp_path / vs.DEFAULT_COLLECTION / "store.json"
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        _store(tmp_path).count()


def test_missing_file_is_an_empty_index_and_reset_clears_the_cache(tmp_path):
    assert _store(tmp_path).count() == 0
    _write_state(tmp_path, 3)
    store = _store(tmp_path)
    assert store.count() == 3
    store.reset()
    assert store.count() == 0
    assert _store(tmp_path).count() == 0
