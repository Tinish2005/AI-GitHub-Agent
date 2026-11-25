"""Loop 2 tests: Execute's generate step gets real repository code.

All tests here are OFFLINE. They use fake embeddings, so they prove the
code is passed on correctly. They do NOT prove real Gemini retrieval quality.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.agent.engine import ExecutionEngine
from backend.agent.executor import (
    REPO_CONTEXT_KEY,
    GenerateExecutor,
    RetrieveExecutor,
    StepContext,
    make_default_executors,
)
from backend.agent.fix_models import FixProposal
from backend.agent.models import Plan, PlanStep, StepKind, StepStatus
from backend.agent.repo_context import format_chunks, retrieve_repo_context
from backend.indexing.ast_parser import parse_source
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.vector_store import VectorStore
from backend.main import (
    _retrieve_repo_context,
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


class _SpyGenerator:
    """Records the context it receives and returns a fixed proposal."""

    model = "spy"

    def __init__(self) -> None:
        self.calls: list = []

    def propose(self, goal: str, context: str) -> FixProposal:
        self.calls.append((goal, context))
        return FixProposal(
            goal=goal, explanation="ok", diff="d", model=self.model, confidence=0.5,
        )


class _CountingStore:
    """A store double that counts queries and returns canned results."""

    def __init__(self, results: list | None = None, *, fail: bool = False) -> None:
        self.results = results or []
        self.fail = fail
        self.queries = 0

    def query(self, text: str, *, top_k: int = 5) -> list:
        self.queries += 1
        if self.fail:
            raise RuntimeError("store is down")
        return self.results


def _hit(qname: str, path: str, code: str, distance: float = 0.1):
    return SimpleNamespace(
        metadata={"qualified_name": qname, "file_path": path},
        document=code,
        distance=distance,
    )


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    svc = EmbeddingService(backend=_FakeBackend())
    s = VectorStore(
        persist_directory=tmp_path / "vdb",
        collection_name=f"ctx_{tmp_path.name}",
        embedding_service=svc,
    )
    s.add_chunks(parse_source(SAMPLE_SOURCE, Path("sample.py")))
    return s


def _ctx(goal: str = "fix the add bug", prior: dict | None = None, kind=StepKind.GENERATE):
    step = PlanStep(id=2, kind=kind, description="step")
    return StepContext(goal=goal, step=step, prior_outputs={} if prior is None else prior)


def _plan(*kinds: StepKind) -> Plan:
    steps = tuple(PlanStep(id=i + 1, kind=k, description=f"step {i + 1}") for i, k in enumerate(kinds))
    return Plan(goal="fix the add bug", steps=steps, strategy="rule_based")


# --- shared helper: repo_context.py ------------------------------------


def test_format_chunks_includes_name_path_and_source() -> None:
    out = format_chunks([_hit("math.add", "src/math.py", "def add(a, b):\n    return a + b")])
    assert out.startswith("[1] math.add (src/math.py)\n")
    assert "return a + b" in out


def test_format_chunks_numbers_and_separates_blocks() -> None:
    out = format_chunks([_hit("a", "a.py", "A"), _hit("b", "b.py", "B")])
    assert out == "[1] a (a.py)\nA\n\n[2] b (b.py)\nB"


def test_retrieve_repo_context_returns_empty_when_nothing_found() -> None:
    assert retrieve_repo_context(_CountingStore([]), "goal") == ""


def test_retrieve_repo_context_returns_empty_when_store_fails() -> None:
    assert retrieve_repo_context(_CountingStore(fail=True), "goal") == ""


def test_main_helper_matches_shared_helper(store: VectorStore) -> None:
    assert _retrieve_repo_context(store, "fix the add bug") == retrieve_repo_context(
        store, "fix the add bug"
    )


# --- RetrieveExecutor keeps the real code ------------------------------


def test_retrieve_saves_real_code_but_summary_stays_short(store: VectorStore) -> None:
    ctx = _ctx(kind=StepKind.RETRIEVE)
    out = RetrieveExecutor(store).run(ctx)
    assert out.startswith("Retrieved 1 chunks:")
    assert "return a + b" not in out
    saved = ctx.prior_outputs[REPO_CONTEXT_KEY]
    assert "def add(a: int, b: int)" in saved
    assert "return a + b" in saved
    assert "sample.py" in saved


def test_retrieve_with_no_results_saves_empty_context() -> None:
    ctx = _ctx(kind=StepKind.RETRIEVE)
    out = RetrieveExecutor(_CountingStore([])).run(ctx)
    assert out == "No matching chunks found for the goal."
    assert ctx.prior_outputs[REPO_CONTEXT_KEY] == ""


# --- GenerateExecutor uses the real code -------------------------------


def test_generate_passes_saved_code_to_generator() -> None:
    spy = _SpyGenerator()
    ctx = _ctx(prior={REPO_CONTEXT_KEY: "[1] add (sample.py)\ncode here", 1: "summary"})
    GenerateExecutor(spy).run(ctx)  # type: ignore[arg-type]
    assert spy.calls == [("fix the add bug", "[1] add (sample.py)\ncode here")]


def test_generate_falls_back_to_store_when_there_was_no_retrieve_step(store: VectorStore) -> None:
    spy = _SpyGenerator()
    GenerateExecutor(spy, vector_store=store).run(_ctx())  # type: ignore[arg-type]
    context = spy.calls[0][1]
    assert "def add(a: int, b: int)" in context
    assert "return a + b" in context


def test_generate_does_not_search_again_after_retrieve_found_nothing() -> None:
    spy = _SpyGenerator()
    counting = _CountingStore([])
    ctx = _ctx(prior={REPO_CONTEXT_KEY: ""})
    out = GenerateExecutor(spy, vector_store=counting).run(ctx)  # type: ignore[arg-type]
    assert counting.queries == 0
    assert "Context: none found" in out


def test_generate_says_so_when_no_code_is_available() -> None:
    spy = _SpyGenerator()
    out = GenerateExecutor(spy).run(_ctx())  # type: ignore[arg-type]
    assert spy.calls[0][1] == ""
    assert "Context: none found, proposal is based on the goal only" in out


def test_generate_reports_when_code_is_attached() -> None:
    spy = _SpyGenerator()
    out = GenerateExecutor(spy).run(_ctx(prior={REPO_CONTEXT_KEY: "some code"}))  # type: ignore[arg-type]
    assert "Context: repository code attached" in out
    assert "Patch (unified diff):" in out


def test_generate_ignores_retrieve_summary_text() -> None:
    """The old bug: the step-1 summary text must never be used as code context."""
    spy = _SpyGenerator()
    ctx = _ctx(prior={1: "Retrieved 5 chunks:\n  1. add  (sample.py)  distance=0.1"})
    GenerateExecutor(spy).run(ctx)  # type: ignore[arg-type]
    assert spy.calls[0][1] == ""


# --- wiring -------------------------------------------------------------


def test_make_default_executors_gives_generate_the_store(store: VectorStore) -> None:
    pipeline = RAGPipeline(vector_store=store, llm=EchoLLMClient())
    executors = make_default_executors(store, pipeline, generator=_SpyGenerator())  # type: ignore[arg-type]
    assert executors[StepKind.GENERATE].vector_store is store


def test_engine_chain_retrieve_then_generate_passes_code(store: VectorStore) -> None:
    spy = _SpyGenerator()
    executors = {
        StepKind.RETRIEVE: RetrieveExecutor(store),
        StepKind.GENERATE: GenerateExecutor(spy, vector_store=store),  # type: ignore[arg-type]
    }
    result = ExecutionEngine(executors).run(_plan(StepKind.RETRIEVE, StepKind.GENERATE))
    assert [s.status for s in result.steps] == [StepStatus.COMPLETED] * 2
    context = spy.calls[0][1]
    assert "def add(a: int, b: int)" in context
    assert "return a + b" in context
    assert "Context: repository code attached" in result.steps[1].output


# --- POST /execute end to end -------------------------------------------


@pytest.fixture
def spy_and_client(store: VectorStore) -> Iterator:
    pipeline = RAGPipeline(vector_store=store, llm=EchoLLMClient())
    spy = _SpyGenerator()
    executors = make_default_executors(store, pipeline, generator=spy)  # type: ignore[arg-type]
    engine = ExecutionEngine(executors=executors)

    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    app.dependency_overrides[get_llm_client] = lambda: EchoLLMClient()
    app.dependency_overrides[get_execution_engine] = lambda: engine

    yield spy, TestClient(app)
    app.dependency_overrides.clear()


def test_execute_route_gives_the_generator_real_code(spy_and_client) -> None:
    spy, client = spy_and_client
    resp = client.post("/execute", json={"goal": "fix the add bug"})
    assert resp.status_code == 200
    assert spy.calls, "the generate step never ran"
    context = spy.calls[0][1]
    assert "def add(a: int, b: int)" in context
    assert "return a + b" in context


def test_execute_context_equals_fix_propose_context(spy_and_client, store: VectorStore) -> None:
    spy, client = spy_and_client
    client.post("/execute", json={"goal": "fix the add bug"})
    assert spy.calls[0][1] == _retrieve_repo_context(store, "fix the add bug")
