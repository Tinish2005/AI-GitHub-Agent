"""Unit tests for backend.agent.executor."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.agent.executor import (
    AnalyzeExecutor,
    GitHubReadExecutor,
    PlannedExecutor,
    RetrieveExecutor,
    StepContext,
    make_default_executors,
)
from backend.agent.models import PlanStep, StepKind
from backend.github.client import FakeGitHubClient
from backend.github.models import Issue, RepoCoord
from backend.indexing.ast_parser import parse_source
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.vector_store import VectorStore
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
def store(tmp_path: Path) -> VectorStore:
    svc = EmbeddingService(backend=_FakeBackend())
    s = VectorStore(
        persist_directory=tmp_path / "vdb",
        collection_name=f"exec_{tmp_path.name}",
        embedding_service=svc,
    )
    s.add_chunks(parse_source(SAMPLE_SOURCE, Path("sample.py")))
    return s


@pytest.fixture
def pipeline(store: VectorStore) -> RAGPipeline:
    return RAGPipeline(vector_store=store, llm=EchoLLMClient())


def _ctx(goal: str = "how to add numbers") -> StepContext:
    step = PlanStep(id=1, kind=StepKind.RETRIEVE, description="do the thing")
    return StepContext(goal=goal, step=step, prior_outputs={})


def test_retrieve_executor_returns_text(store: VectorStore) -> None:
    ex = RetrieveExecutor(store, top_k=3)
    out = ex.run(_ctx())
    assert "Retrieved" in out


def test_retrieve_executor_rejects_bad_top_k(store: VectorStore) -> None:
    with pytest.raises(ValueError):
        RetrieveExecutor(store, top_k=0)


def test_analyze_executor_returns_answer(pipeline: RAGPipeline) -> None:
    ex = AnalyzeExecutor(pipeline)
    out = ex.run(_ctx())
    assert "sources:" in out


def test_github_read_executor_no_coord_returns_message() -> None:
    ex = GitHubReadExecutor(FakeGitHubClient(), coord=None)
    out = ex.run(_ctx())
    assert "no repository was configured" in out.lower()


def test_github_read_executor_lists_issues() -> None:
    coord = RepoCoord(owner="o", repo="r")
    fake = FakeGitHubClient()
    fake.add_issue(coord, Issue(
        number=1, title="Bug A", state="open", author="alice", url="u",
    ))
    ex = GitHubReadExecutor(fake, coord=coord)
    out = ex.run(_ctx())
    assert "Bug A" in out
    assert "o/r" in out


def test_planned_executor_reports_note() -> None:
    ex = PlannedExecutor(StepKind.VALIDATE, "later")
    out = ex.run(_ctx())
    assert "[planned]" in out
    assert "validate" in out


def test_make_default_executors_covers_all_kinds(
    store: VectorStore, pipeline: RAGPipeline,
) -> None:
    executors = make_default_executors(store, pipeline, github=None)
    for k in StepKind:
        assert k in executors


def test_make_default_executors_uses_github_when_provided(
    store: VectorStore, pipeline: RAGPipeline,
) -> None:
    coord = RepoCoord(owner="o", repo="r")
    fake = FakeGitHubClient()
    executors = make_default_executors(store, pipeline, github=fake, coord=coord)
    assert isinstance(executors[StepKind.GITHUB_READ], GitHubReadExecutor)