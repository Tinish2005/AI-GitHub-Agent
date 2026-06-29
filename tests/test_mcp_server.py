"""Integration tests for `backend.mcp.server`."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.indexing.ast_parser import parse_source
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.vector_store import VectorStore
from backend.mcp.models import ErrorCode, JsonRpcRequest
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
def server(tmp_path: Path) -> MCPServer:
    svc = EmbeddingService(backend=_FakeBackend())
    store = VectorStore(
        persist_directory=tmp_path / "vdb",
        collection_name=f"srv_{tmp_path.name}",
        embedding_service=svc,
    )
    store.add_chunks(parse_source(SAMPLE_SOURCE, Path("sample.py")))
    pipeline = RAGPipeline(vector_store=store, llm=EchoLLMClient())
    registry = build_default_registry(store, pipeline)
    return MCPServer(registry=registry)


def test_initialize_returns_protocol_info(server: MCPServer) -> None:
    resp = server.handle(JsonRpcRequest(id=1, method="initialize"))
    assert resp.error is None
    assert resp.result is not None
    assert resp.result["protocolVersion"]
    assert resp.result["serverName"] == "ai-github-agent"


def test_tools_list_returns_three_tools(server: MCPServer) -> None:
    resp = server.handle(JsonRpcRequest(id=2, method="tools/list"))
    assert resp.error is None
    assert resp.result is not None
    names = [t["name"] for t in resp.result["tools"]]
    assert set(names) == {"search_code", "get_chunk", "ask_codebase"}


def test_tools_call_search_code(server: MCPServer) -> None:
    resp = server.handle(
        JsonRpcRequest(
            id=3,
            method="tools/call",
            params={"name": "search_code", "arguments": {"query": "add"}},
        )
    )
    assert resp.error is None
    assert resp.result is not None
    assert resp.result["isError"] is False
    assert resp.result["content"][0]["type"] == "text"


def test_tools_call_unknown_tool_returns_is_error(server: MCPServer) -> None:
    resp = server.handle(
        JsonRpcRequest(
            id=4,
            method="tools/call",
            params={"name": "does_not_exist", "arguments": {}},
        )
    )
    assert resp.error is None  # top-level still OK
    assert resp.result is not None
    assert resp.result["isError"] is True


def test_tools_call_missing_name(server: MCPServer) -> None:
    resp = server.handle(
        JsonRpcRequest(id=5, method="tools/call", params={"arguments": {}})
    )
    assert resp.error is not None
    assert resp.error.code == ErrorCode.INVALID_PARAMS


def test_unknown_method_returns_method_not_found(server: MCPServer) -> None:
    resp = server.handle(JsonRpcRequest(id=6, method="nope"))
    assert resp.error is not None
    assert resp.error.code == ErrorCode.METHOD_NOT_FOUND


def test_response_carries_request_id(server: MCPServer) -> None:
    resp = server.handle(JsonRpcRequest(id="abc", method="initialize"))
    assert resp.id == "abc"