"""Unit tests for the index_repo MCP tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.cloning.cloner import FakeCloner
from backend.cloning.indexing_service import IndexingService
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.vector_store import VectorStore
from backend.mcp.tools import make_index_repo_tool


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
def indexer(tmp_path: Path) -> IndexingService:
    svc = EmbeddingService(backend=_FakeBackend())
    store = VectorStore(
        persist_directory=tmp_path / "vdb",
        collection_name=f"tool_{tmp_path.name}",
        embedding_service=svc,
    )
    cloner = FakeCloner(cache_dir=tmp_path / "cache")
    return IndexingService(cloner=cloner, vector_store=store)


def test_index_repo_tool_returns_summary(indexer: IndexingService) -> None:
    tool = make_index_repo_tool(indexer)
    out = tool.execute({"url": "https://github.com/fake/proj.git"})
    assert "Indexed" in out
    assert "Chunks indexed:" in out
    assert "Cached clone: no" in out


def test_index_repo_tool_requires_url(indexer: IndexingService) -> None:
    tool = make_index_repo_tool(indexer)
    with pytest.raises(ValueError):
        tool.execute({})
    with pytest.raises(ValueError):
        tool.execute({"url": "  "})


def test_index_repo_tool_supports_force(indexer: IndexingService) -> None:
    tool = make_index_repo_tool(indexer)
    tool.execute({"url": "https://github.com/fake/proj.git"})
    out2 = tool.execute({"url": "https://github.com/fake/proj.git", "force": True})
    assert "Cached clone: no" in out2