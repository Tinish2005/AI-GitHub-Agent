"""Unit tests for `backend.mcp.tools`."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.indexing.ast_parser import parse_source
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.vector_store import VectorStore
from backend.mcp.tools import (
    Tool,
    ToolRegistry,
    build_default_registry,
    make_ask_codebase_tool,
    make_get_chunk_tool,
    make_search_code_tool,
)
from backend.rag.llm import EchoLLMClient
from backend.rag.pipeline import RAGPipeline


SAMPLE_SOURCE = '''\
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}"
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
def store(tmp_path: Path) -> VectorStore:
    svc = EmbeddingService(backend=_FakeBackend())
    s = VectorStore(
        persist_directory=tmp_path / "vdb",
        collection_name=f"mcp_{tmp_path.name}",
        embedding_service=svc,
    )
    s.add_chunks(parse_source(SAMPLE_SOURCE, Path("sample.py")))
    return s


@pytest.fixture
def pipeline(store: VectorStore) -> RAGPipeline:
    return RAGPipeline(vector_store=store, llm=EchoLLMClient())


# ----- Registry -----


def test_registry_register_and_get() -> None:
    reg = ToolRegistry()
    t = Tool(name="x", description="d", input_schema={}, execute=lambda p: "ok")
    reg.register(t)
    assert reg.get("x") is t
    assert reg.get("nope") is None


def test_registry_rejects_duplicate_names() -> None:
    reg = ToolRegistry()
    t1 = Tool(name="x", description="d", input_schema={}, execute=lambda p: "1")
    t2 = Tool(name="x", description="d", input_schema={}, execute=lambda p: "2")
    reg.register(t1)
    with pytest.raises(ValueError):
        reg.register(t2)


def test_registry_lists_alphabetically() -> None:
    reg = ToolRegistry()
    reg.register(Tool(name="b", description="d", input_schema={}, execute=lambda p: ""))
    reg.register(Tool(name="a", description="d", input_schema={}, execute=lambda p: ""))
    assert [t.name for t in reg.list_tools()] == ["a", "b"]


# ----- search_code -----


def test_search_code_tool_returns_text(store: VectorStore) -> None:
    tool = make_search_code_tool(store)
    out = tool.execute({"query": "add numbers", "top_k": 2})
    assert "matches" in out.lower() or "found" in out.lower()


def test_search_code_requires_query(store: VectorStore) -> None:
    tool = make_search_code_tool(store)
    with pytest.raises(ValueError):
        tool.execute({})


def test_search_code_rejects_bad_top_k(store: VectorStore) -> None:
    tool = make_search_code_tool(store)
    with pytest.raises(ValueError):
        tool.execute({"query": "x", "top_k": 0})


# ----- get_chunk -----


def test_get_chunk_returns_document(store: VectorStore) -> None:
    chunks = parse_source(SAMPLE_SOURCE, Path("sample.py"))
    ids = store.add_chunks(chunks)
    tool = make_get_chunk_tool(store)
    out = tool.execute({"chunk_id": ids[0]})
    assert "def " in out  # contains source


def test_get_chunk_unknown_id(store: VectorStore) -> None:
    tool = make_get_chunk_tool(store)
    with pytest.raises(ValueError):
        tool.execute({"chunk_id": "deadbeef"})


# ----- ask_codebase -----


def test_ask_codebase_returns_answer_with_sources(pipeline: RAGPipeline) -> None:
    tool = make_ask_codebase_tool(pipeline)
    out = tool.execute({"question": "how do I add?", "top_k": 2})
    assert "Sources:" in out
    assert "ECHO" in out  # EchoLLMClient signature


def test_ask_codebase_requires_question(pipeline: RAGPipeline) -> None:
    tool = make_ask_codebase_tool(pipeline)
    with pytest.raises(ValueError):
        tool.execute({})


# ----- default registry -----


def test_build_default_registry_has_three_tools(
    store: VectorStore, pipeline: RAGPipeline
) -> None:
    reg = build_default_registry(store, pipeline)
    names = [t.name for t in reg.list_tools()]
    assert names == ["ask_codebase", "get_chunk", "search_code"]