"""Integration tests for the `POST /qa` endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.indexing.ast_parser import parse_source
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.vector_store import VectorStore
from backend.main import app, get_rag_pipeline, get_vector_store
from backend.rag.llm import EchoLLMClient
from backend.rag.pipeline import RAGPipeline


SAMPLE_SOURCE = '''\
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''


class _FakeBackend:
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[list[float]]:
        return [[float(len(s)) + i for i in range(EMBEDDING_DIMENSION)] for s in sentences]


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A TestClient with a fully-faked RAG pipeline (no OpenAI calls)."""
    svc = EmbeddingService(backend=_FakeBackend())
    store = VectorStore(
        persist_directory=tmp_path / "vdb",
        collection_name="qa_test",
        embedding_service=svc,
    )
    store.add_chunks(parse_source(SAMPLE_SOURCE, Path("sample.py")))

    def _fake_pipeline() -> RAGPipeline:
        return RAGPipeline(vector_store=store, llm=EchoLLMClient())

    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_rag_pipeline] = _fake_pipeline

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_qa_endpoint_returns_200(client: TestClient) -> None:
    resp = client.post("/qa", json={"question": "How do I add?"})
    assert resp.status_code == 200


def test_qa_endpoint_returns_answer_shape(client: TestClient) -> None:
    resp = client.post("/qa", json={"question": "How do I add?"})
    body = resp.json()
    assert body["question"] == "How do I add?"
    assert body["answer"]
    assert body["model"] == "echo-test"
    assert isinstance(body["sources"], list)


def test_qa_endpoint_rejects_empty_question(client: TestClient) -> None:
    resp = client.post("/qa", json={"question": ""})
    assert resp.status_code in (400, 422)  # 422 from pydantic, 400 from us


def test_qa_endpoint_respects_top_k(client: TestClient) -> None:
    resp = client.post("/qa", json={"question": "add", "top_k": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sources"]) <= 1