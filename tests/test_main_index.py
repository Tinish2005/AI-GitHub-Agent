"""Integration tests for the /index endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.cloning.cloner import FakeCloner
from backend.cloning.indexing_service import IndexingService
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.vector_store import VectorStore
from backend.main import app, get_indexing_service


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
def client(tmp_path: Path):
    svc = EmbeddingService(backend=_FakeBackend())
    store = VectorStore(
        persist_directory=tmp_path / "vdb",
        collection_name="index_route",
        embedding_service=svc,
    )
    cloner = FakeCloner(cache_dir=tmp_path / "cache")
    indexer = IndexingService(cloner=cloner, vector_store=store)

    app.dependency_overrides[get_indexing_service] = lambda: indexer
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_index_endpoint_returns_200(client: TestClient) -> None:
    resp = client.post("/index", json={"url": "https://github.com/fake/proj.git"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["url"] == "https://github.com/fake/proj.git"
    assert body["chunks_indexed"] > 0
    assert body["was_cached"] is False


def test_index_endpoint_second_call_is_cached(client: TestClient) -> None:
    client.post("/index", json={"url": "https://github.com/fake/proj.git"})
    resp = client.post("/index", json={"url": "https://github.com/fake/proj.git"})
    body = resp.json()
    assert body["was_cached"] is True


def test_index_endpoint_rejects_empty_url(client: TestClient) -> None:
    resp = client.post("/index", json={"url": ""})
    assert resp.status_code in (400, 422)


def test_index_endpoint_supports_force(client: TestClient) -> None:
    client.post("/index", json={"url": "https://github.com/fake/proj.git"})
    resp = client.post(
        "/index",
        json={"url": "https://github.com/fake/proj.git", "force": True},
    )
    body = resp.json()
    assert body["was_cached"] is False