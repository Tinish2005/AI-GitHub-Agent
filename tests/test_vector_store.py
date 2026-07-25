"""Integration tests for `backend.indexing.vector_store`."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.indexing.ast_parser import parse_source
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.metadata import extract_metadata_for_chunks
from backend.indexing.vector_store import (
    DEFAULT_COLLECTION,
    QueryResult,
    VectorStore,
)


SAMPLE_SOURCE = '''\
"""Sample module."""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}"


class Calculator:
    """A tiny calculator."""

    def multiply(self, a: int, b: int) -> int:
        return a * b
'''


class _FakeBackend:
    """Length-based deterministic encoder for fast tests."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[list[float]]:
        out: list[list[float]] = []
        for s in sentences:
            base = float(len(s))
            out.append([base + i for i in range(EMBEDDING_DIMENSION)])
        return out


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    """A fresh VectorStore backed by a temp directory + fake embeddings."""
    svc = EmbeddingService(backend=_FakeBackend())
    return VectorStore(
        persist_directory=tmp_path / "vdb",
        collection_name=f"test_{tmp_path.name}",
        embedding_service=svc,
    )


def test_default_collection_constant() -> None:
    assert DEFAULT_COLLECTION == "code_chunks"


def test_add_chunks_returns_one_id_per_chunk(store: VectorStore) -> None:
    chunks = parse_source(SAMPLE_SOURCE, Path("sample.py"))
    ids = store.add_chunks(chunks)
    assert len(ids) == len(chunks)
    assert all(isinstance(i, str) and len(i) == 64 for i in ids)


def test_add_chunks_persists_count(store: VectorStore) -> None:
    chunks = parse_source(SAMPLE_SOURCE, Path("sample.py"))
    store.add_chunks(chunks)
    assert store.count() == len(chunks)


def test_add_chunks_is_idempotent(store: VectorStore) -> None:
    chunks = parse_source(SAMPLE_SOURCE, Path("sample.py"))
    store.add_chunks(chunks)
    store.add_chunks(chunks)  # upsert, not append
    assert store.count() == len(chunks)


def test_add_chunks_with_metadata(store: VectorStore, tmp_path: Path) -> None:
    f = tmp_path / "sample.py"
    f.write_text(SAMPLE_SOURCE, encoding="utf-8")
    chunks = parse_source(SAMPLE_SOURCE, f)
    metas = extract_metadata_for_chunks(chunks, project_root=tmp_path)
    ids = store.add_chunks(chunks, metas)
    assert ids == [m.content_hash for m in metas]


def test_add_chunks_rejects_mismatched_metadata(store: VectorStore) -> None:
    chunks = parse_source(SAMPLE_SOURCE, Path("sample.py"))
    metas = extract_metadata_for_chunks(chunks[:1])
    with pytest.raises(ValueError):
        store.add_chunks(chunks, metas)


def test_add_chunks_empty_list_is_noop(store: VectorStore) -> None:
    assert store.add_chunks([]) == []
    assert store.count() == 0


def test_query_returns_query_results(store: VectorStore) -> None:
    chunks = parse_source(SAMPLE_SOURCE, Path("sample.py"))
    store.add_chunks(chunks)
    results = store.query("how do I add two numbers", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    assert all(isinstance(r, QueryResult) for r in results)


def test_query_results_have_required_fields(store: VectorStore) -> None:
    chunks = parse_source(SAMPLE_SOURCE, Path("sample.py"))
    store.add_chunks(chunks)
    results = store.query("add", top_k=1)
    assert results
    r = results[0]
    assert r.chunk_id
    assert r.document
    assert isinstance(r.metadata, dict)
    assert "qualified_name" in r.metadata
    assert isinstance(r.distance, float)


def test_query_rejects_empty_text(store: VectorStore) -> None:
    with pytest.raises(ValueError):
        store.query("", top_k=3)


def test_query_rejects_zero_top_k(store: VectorStore) -> None:
    with pytest.raises(ValueError):
        store.query("hello", top_k=0)


def test_reset_clears_collection(store: VectorStore) -> None:
    chunks = parse_source(SAMPLE_SOURCE, Path("sample.py"))
    store.add_chunks(chunks)
    assert store.count() > 0
    store.reset()
    assert store.count() == 0