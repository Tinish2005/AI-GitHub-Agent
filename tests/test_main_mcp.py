"""Integration tests for the `POST /mcp` endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.indexing.ast_parser import parse_source
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.vector_store import VectorStore
from backend.main import app, get_mcp_server, get_rag_pipeline, get_vector_store
from backend.mcp.server import MCPServer
from backend.mcp.tools import build_default_registry
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
    svc = EmbeddingService(backend=_FakeBackend())
    store = VectorStore(
        persist_directory=tmp_path / "vdb",
        collection_name="mcp_main",
        embedding_service=svc,
    )
    store.add_chunks(parse_source(SAMPLE_SOURCE, Path("sample.py")))
    pipeline = RAGPipeline(vector_store=store, llm=EchoLLMClient())
    server = MCPServer(registry=build_default_registry(store, pipeline))

    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    app.dependency_overrides[get_mcp_server] = lambda: server

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_mcp_initialize_round_trip(client: TestClient) -> None:
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == 1
    assert body["result"]["protocolVersion"]


def test_mcp_tools_list_round_trip(client: TestClient) -> None:
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )
    assert resp.status_code == 200
    body = resp.json()
    names = [t["name"] for t in body["result"]["tools"]]
    assert set(names) == {"search_code", "get_chunk", "ask_codebase"}


def test_mcp_tools_call_round_trip(client: TestClient) -> None:
    resp = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search_code", "arguments": {"query": "add"}},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["isError"] is False


def test_mcp_invalid_payload_returns_400(client: TestClient) -> None:
    resp = client.post("/mcp", json={"not": "a valid request"})
    assert resp.status_code == 400