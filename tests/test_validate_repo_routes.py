"""Loop 3 tests: the repo option on /fix/validate, /fix/pr and /execute (OFFLINE)."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.agent.draft_pr_service import FakeDraftPRService
from backend.agent.engine import ExecutionEngine
from backend.agent.executor import (
    REPO_TARGET_KEY,
    GenerateExecutor,
    StepContext,
    ValidateExecutor,
    make_default_executors,
)
from backend.agent.fix_generator import FixGenerator
from backend.agent.fix_models import FixHunk, FixProposal
from backend.agent.models import Plan, PlanStep, StepKind, StepStatus
from backend.agent.validation_checks import SyntaxCheck
from backend.agent.validation_models import RepoTarget
from backend.agent.validation_pipeline import ValidationPipeline
from backend.config import Settings, get_settings
from backend.indexing.ast_parser import parse_source
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.vector_store import VectorStore
from backend.main import (
    app,
    get_draft_pr_service,
    get_execution_engine,
    get_fix_generator,
    get_llm_client,
    get_rag_pipeline,
    get_validation_pipeline,
    get_vector_store,
)
from backend.rag.llm import EchoLLMClient
from backend.rag.pipeline import RAGPipeline
from tests.test_validation_real_repo import CALC, CALC_FIXED, COMMIT, FakeProvider


def _diff(path: str, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    ))


class _StaticLLM:
    def __init__(self, diff: str) -> None:
        self._response = json.dumps({"explanation": "fix add", "diff": diff, "confidence": 0.9})

    @property
    def model(self) -> str:
        return "static-fake"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


GOOD = _diff("pkg/calc.py", CALC, CALC_FIXED)
BAD = _diff("pkg/calc.py", CALC.replace("a - b", "a * b"), CALC)


def _setup(tmp_path: Path, diff: str):
    provider = FakeProvider({"pkg/calc.py": CALC})
    validator = ValidationPipeline(
        checks=(SyntaxCheck(),), sandbox_parent=tmp_path / "boxes", snapshot_provider=provider,
    )
    gen = FixGenerator(llm=_StaticLLM(diff))
    app.dependency_overrides[get_fix_generator] = lambda: gen
    app.dependency_overrides[get_validation_pipeline] = lambda: validator
    return provider, TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator:
    yield
    app.dependency_overrides.clear()


# --- /fix/validate -------------------------------------------------------


def test_validate_route_checks_the_patch_against_the_named_repo(tmp_path: Path) -> None:
    provider, client = _setup(tmp_path, GOOD)
    resp = client.post("/fix/validate", json={
        "goal": "fix add", "context": "x", "owner": "o", "repo": "r", "ref": "main",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is True
    assert body["base_commit"] == COMMIT
    assert "o/r@abc1234" in body["base_description"]
    assert provider.calls[0]["ref"] == "main"


def test_validate_route_reports_the_file_and_reason_for_a_bad_patch(tmp_path: Path) -> None:
    _, client = _setup(tmp_path, BAD)
    body = client.post("/fix/validate", json={
        "goal": "fix add", "context": "x", "owner": "o", "repo": "r",
    }).json()
    assert body["passed"] is False
    assert "pkg/calc.py" in body["error"]
    assert "does not match the real file" in body["error"]


def test_validate_route_without_a_repo_still_works_and_says_so(tmp_path: Path) -> None:
    new_file = "--- /dev/null\n+++ b/hello.py\n@@ -0,0 +1 @@\n+x = 1\n"
    provider, client = _setup(tmp_path, new_file)
    body = client.post("/fix/validate", json={"goal": "fix add", "context": "x"}).json()
    assert body["passed"] is True
    assert body["base_description"] == "empty folder (no repository was given)"
    assert provider.calls == []


@pytest.mark.parametrize("payload", [{"owner": "o"}, {"repo": "r"}])
def test_validate_route_needs_owner_and_repo_together(tmp_path: Path, payload: dict) -> None:
    _, client = _setup(tmp_path, GOOD)
    resp = client.post("/fix/validate", json={"goal": "g", "context": "x", **payload})
    assert resp.status_code == 422


def test_validate_route_reports_a_fetch_failure_clearly(tmp_path: Path) -> None:
    from backend.github.repo_snapshot import SnapshotError

    provider, client = _setup(tmp_path, GOOD)
    provider.fail = SnapshotError("Not found on GitHub: /repos/o/r")
    body = client.post("/fix/validate", json={
        "goal": "g", "context": "x", "owner": "o", "repo": "r",
    }).json()
    assert body["passed"] is False
    assert "Could not get the repository files" in body["error"]


# --- /fix/pr -------------------------------------------------------------


def test_pr_route_validates_against_the_repo_and_branch_in_the_request(tmp_path: Path) -> None:
    provider, client = _setup(tmp_path, GOOD)
    svc = FakeDraftPRService(next_number=7)
    app.dependency_overrides[get_draft_pr_service] = lambda: svc
    # These tests send only a goal, so they use the escape hatch on purpose.
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, allow_regenerated_pr=True)
    resp = client.post("/fix/pr", json={
        "goal": "fix add", "context": "x", "owner": "o", "repo": "r", "base_branch": "dev",
    })
    assert resp.status_code == 200
    assert resp.json()["created"] is True
    assert provider.calls[0]["owner"] == "o"
    assert provider.calls[0]["ref"] == "dev"


def test_pr_route_does_not_open_a_pr_for_a_patch_that_does_not_fit(tmp_path: Path) -> None:
    _, client = _setup(tmp_path, BAD)
    app.dependency_overrides[get_draft_pr_service] = lambda: FakeDraftPRService()
    # These tests send only a goal, so they use the escape hatch on purpose.
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, allow_regenerated_pr=True)
    body = client.post("/fix/pr", json={
        "goal": "fix add", "context": "x", "owner": "o", "repo": "r",
    }).json()
    assert body["created"] is False


# --- engine and executor -------------------------------------------------


class _Recorder:
    kind = StepKind.RETRIEVE

    def __init__(self) -> None:
        self.seen = None

    def run(self, context: StepContext) -> str:
        self.seen = context.prior_outputs.get(REPO_TARGET_KEY)
        return "ok"


def test_engine_seeds_the_shared_data_without_changing_the_callers_dict() -> None:
    rec = _Recorder()
    plan = Plan(goal="g", steps=(PlanStep(id=1, kind=StepKind.RETRIEVE, description="d"),))
    seed = {REPO_TARGET_KEY: "target"}
    ExecutionEngine({StepKind.RETRIEVE: rec}).run(plan, initial_outputs=seed)
    assert rec.seen == "target"
    assert seed == {REPO_TARGET_KEY: "target"}


def test_engine_run_without_seed_still_works() -> None:
    rec = _Recorder()
    plan = Plan(goal="g", steps=(PlanStep(id=1, kind=StepKind.RETRIEVE, description="d"),))
    ExecutionEngine({StepKind.RETRIEVE: rec}).run(plan)
    assert rec.seen is None


class _PipelineSpy:
    """A validator whose validate() has no repo parameter (older style)."""

    def __init__(self) -> None:
        self.calls: list = []

    def validate(self, proposal, **kwargs):
        self.calls.append(kwargs)
        from backend.agent.validation_models import ValidationResult
        return ValidationResult(proposal_goal=proposal.goal, passed=True, score=1.0)


def _proposal() -> FixProposal:
    return FixProposal(
        goal="g", explanation="e", diff="d", hunks=(FixHunk(file_path="a.py", added_lines=1, removed_lines=0),),
        model="m", confidence=0.5, is_valid=True,
    )


def test_validate_executor_passes_the_repo_only_when_one_is_given() -> None:
    spy = _PipelineSpy()
    step = PlanStep(id=2, kind=StepKind.VALIDATE, description="v")
    target = RepoTarget(owner="o", repo="r")
    ValidateExecutor(spy).run(StepContext(  # type: ignore[arg-type]
        goal="g", step=step, prior_outputs={"fix_proposal": _proposal(), REPO_TARGET_KEY: target},
    ))
    ValidateExecutor(spy).run(StepContext(  # type: ignore[arg-type]
        goal="g", step=step, prior_outputs={"fix_proposal": _proposal()},
    ))
    assert spy.calls == [{"repo": target}, {}]


# --- /execute end to end ---------------------------------------------------


class _FakeBackend:
    def encode(self, sentences: list, **kwargs) -> list:
        return [[float(len(s)) + i for i in range(EMBEDDING_DIMENSION)] for s in sentences]


def _execute_client(tmp_path: Path, diff: str):
    store = VectorStore(
        persist_directory=tmp_path / "vdb", collection_name=f"v_{tmp_path.name}",
        embedding_service=EmbeddingService(backend=_FakeBackend()),
    )
    store.add_chunks(parse_source(CALC, Path("pkg/calc.py")))
    rag = RAGPipeline(vector_store=store, llm=EchoLLMClient())
    provider = FakeProvider({"pkg/calc.py": CALC})
    validator = ValidationPipeline(
        checks=(SyntaxCheck(),), sandbox_parent=tmp_path / "boxes", snapshot_provider=provider,
    )
    executors = make_default_executors(
        store, rag, generator=FixGenerator(llm=_StaticLLM(diff)), validation_pipeline=validator,
    )
    engine = ExecutionEngine(executors=executors)
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_rag_pipeline] = lambda: rag
    app.dependency_overrides[get_llm_client] = lambda: EchoLLMClient()
    app.dependency_overrides[get_execution_engine] = lambda: engine
    return provider, TestClient(app)


def _validate_step(body: dict) -> dict:
    return next(s for s in body["steps"] if s["kind"] == "validate")


def test_execute_route_validates_the_fix_against_the_named_repo(tmp_path: Path) -> None:
    provider, client = _execute_client(tmp_path, GOOD)
    body = client.post("/execute", json={
        "goal": "fix the add bug", "owner": "o", "repo": "r", "ref": "main",
    }).json()
    step = _validate_step(body)
    assert step["status"] == StepStatus.COMPLETED.value
    assert "passed=True" in step["output"]
    assert "Checked against: o/r@abc1234" in step["output"]
    assert provider.calls[0]["paths"] == ["pkg/calc.py"]
    assert provider.calls[0]["ref"] == "main"


def test_execute_route_shows_the_file_and_reason_for_a_bad_patch(tmp_path: Path) -> None:
    _, client = _execute_client(tmp_path, BAD)
    body = client.post("/execute", json={
        "goal": "fix the add bug", "owner": "o", "repo": "r",
    }).json()
    step = _validate_step(body)
    assert "passed=False" in step["output"]
    assert "pkg/calc.py" in step["output"]
    assert "does not match the real file" in step["output"]


def test_execute_route_without_a_repo_is_unchanged_and_honest(tmp_path: Path) -> None:
    provider, client = _execute_client(tmp_path, GOOD)
    body = client.post("/execute", json={"goal": "fix the add bug"}).json()
    step = _validate_step(body)
    assert provider.calls == []
    assert "empty folder (no repository was given)" in step["output"]


def test_execute_route_needs_owner_and_repo_together(tmp_path: Path) -> None:
    _, client = _execute_client(tmp_path, GOOD)
    resp = client.post("/execute", json={"goal": "fix the add bug", "owner": "o"})
    assert resp.status_code == 422
