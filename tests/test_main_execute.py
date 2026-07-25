"""Integration tests for the /execute endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.agent.engine import ExecutionEngine
from backend.agent.executor import make_default_executors
from backend.indexing.ast_parser import parse_source
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.vector_store import VectorStore
from backend.main import (
    app,
    get_execution_engine,
    get_llm_client,
    get_rag_pipeline,
    get_vector_store,
)
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
        collection_name="exec_route",
        embedding_service=svc,
    )
    store.add_chunks(parse_source(SAMPLE_SOURCE, Path("sample.py")))
    pipeline = RAGPipeline(vector_store=store, llm=EchoLLMClient())
    executors = make_default_executors(store, pipeline)
    engine = ExecutionEngine(executors=executors)

    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    app.dependency_overrides[get_llm_client] = lambda: EchoLLMClient()
    app.dependency_overrides[get_execution_engine] = lambda: engine

    yield TestClient(app)
    app.dependency_overrides.clear()


def test_execute_endpoint_returns_200(client: TestClient) -> None:
    resp = client.post("/execute", json={"goal": "review the auth module"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["goal"] == "review the auth module"
    assert body["steps"]


def test_execute_endpoint_reports_counts(client: TestClient) -> None:
    resp = client.post("/execute", json={"goal": "review the auth module"})
    body = resp.json()
    assert body["completed"] >= 1
    assert body["failed"] >= 0
    assert body["aborted"] is False


def test_execute_endpoint_rejects_empty_goal(client: TestClient) -> None:
    resp = client.post("/execute", json={"goal": ""})
    assert resp.status_code in (400, 422)


def test_execute_endpoint_rejects_unknown_strategy(client: TestClient) -> None:
    resp = client.post("/execute", json={"goal": "do stuff", "strategy": "psychic"})
    assert resp.status_code == 400


def test_execute_endpoint_supports_abort_flag(client: TestClient) -> None:
    resp = client.post("/execute", json={
        "goal": "review auth module",
        "abort_on_failure": False,
    })
    assert resp.status_code == 200