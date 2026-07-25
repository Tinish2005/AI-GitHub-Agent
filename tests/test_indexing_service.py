"""Integration tests for the IndexingService."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.cloning.cloner import FakeCloner
from backend.cloning.indexing_service import IndexingService, IndexResult
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.vector_store import VectorStore


class _FakeBackend:
    def encode(
        self,
        sentences: list,
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list:
        return [[float(len(s)) + i for i in range(EMBEDDING_DIMENSION)] for s in sentences]


@pytest.fixture
def service(tmp_path: Path) -> IndexingService:
    svc = EmbeddingService(backend=_FakeBackend())
    store = VectorStore(
        persist_directory=tmp_path / "vdb",
        collection_name=f"idx_{tmp_path.name}",
        embedding_service=svc,
    )
    cloner = FakeCloner(cache_dir=tmp_path / "cache")
    return IndexingService(cloner=cloner, vector_store=store)


def test_index_repo_returns_result(service: IndexingService) -> None:
    result = service.index_repo("https://github.com/fake/proj.git")
    assert isinstance(result, IndexResult)
    assert result.chunks_indexed > 0
    assert result.files_scanned >= 1
    assert result.was_cached is False


def test_index_repo_persists_chunks(service: IndexingService) -> None:
    service.index_repo("https://github.com/fake/proj.git")
    assert service.vector_store.count() > 0


def test_index_repo_second_call_is_cached(service: IndexingService) -> None:
    r1 = service.index_repo("https://github.com/fake/proj.git")
    r2 = service.index_repo("https://github.com/fake/proj.git")
    assert r1.was_cached is False
    assert r2.was_cached is True


def test_index_repo_force_reindexes(service: IndexingService) -> None:
    service.index_repo("https://github.com/fake/proj.git")
    r2 = service.index_repo("https://github.com/fake/proj.git", force=True)
    assert r2.was_cached is False


def test_index_repo_rejects_empty_url(service: IndexingService) -> None:
    with pytest.raises(ValueError):
        service.index_repo("")


def test_index_repo_finds_functions(service: IndexingService, tmp_path: Path) -> None:
    """The default fake repo has 2 functions - they should end up in the store."""
    service.index_repo("https://github.com/fake/proj.git")
    results = service.vector_store.query("hello function", top_k=5)
    assert len(results) >= 1