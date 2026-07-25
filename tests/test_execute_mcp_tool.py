"""Unit tests for the execute_plan MCP tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.agent.engine import ExecutionEngine
from backend.agent.executor import make_default_executors
from backend.agent.planner import RuleBasedPlanner
from backend.indexing.ast_parser import parse_source
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.vector_store import VectorStore
from backend.mcp.tools import make_execute_plan_tool
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
def engine(tmp_path: Path) -> ExecutionEngine:
    svc = EmbeddingService(backend=_FakeBackend())
    store = VectorStore(
        persist_directory=tmp_path / "vdb",
        collection_name=f"exec_tool_{tmp_path.name}",
        embedding_service=svc,
    )
    store.add_chunks(parse_source(SAMPLE_SOURCE, Path("sample.py")))
    pipeline = RAGPipeline(vector_store=store, llm=EchoLLMClient())
    executors = make_default_executors(store, pipeline)
    return ExecutionEngine(executors=executors)


def test_execute_plan_tool_returns_summary(engine: ExecutionEngine) -> None:
    tool = make_execute_plan_tool(RuleBasedPlanner(), engine)
    out = tool.execute({"goal": "review the auth module"})
    assert "Goal:" in out
    assert "Completed:" in out
    assert "step 1" in out


def test_execute_plan_tool_requires_goal(engine: ExecutionEngine) -> None:
    tool = make_execute_plan_tool(RuleBasedPlanner(), engine)
    with pytest.raises(ValueError):
        tool.execute({})
    with pytest.raises(ValueError):
        tool.execute({"goal": "   "})


def test_execute_plan_tool_reports_planned_steps(engine: ExecutionEngine) -> None:
    """Steps that Loops 11-13 will implement should show as [planned] in the output."""
    tool = make_execute_plan_tool(RuleBasedPlanner(), engine)
    out = tool.execute({"goal": "Fix the login bug"})
    assert "planned" in out.lower() or "generate" in out.lower()