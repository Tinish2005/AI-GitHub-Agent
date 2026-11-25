"""Loop 5 tests: the planner picks a plan that fits the goal, with honest steps (OFFLINE)."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.agent.executor import REPO_TARGET_KEY, GitHubReadExecutor, StepContext, StepSkipped
from backend.agent.models import PlanStep, StepKind
from backend.agent.planner import (
    PLANNER_SYSTEM_PROMPT,
    LLMPlanner,
    RuleBasedPlanner,
)
from backend.agent.validation_models import RepoTarget
from backend.main import app, get_planner
from tests.test_validate_repo_routes import GOOD, _execute_client

R, A, G, V, H = (
    StepKind.RETRIEVE, StepKind.ANALYZE, StepKind.GENERATE, StepKind.VALIDATE, StepKind.GITHUB_READ,
)
RUNNABLE = {R, A, H, G, V}

FIX = (R, A, G, V)
FIX_GITHUB = (R, H, A, G, V)

CASES = [
    # (goal, expected step kinds)
    ("refactor request handling", FIX),
    ("Refactor request handling.", FIX),
    ("optimize request retries", FIX),
    ("fix the login bug", FIX),
    ("Fix: sessions crash on redirect!", FIX),
    ("login bugs keep crashing the app", FIX),
    ("the upload failed with an error", FIX),
    ("add exponential backoff to retries", FIX),
    ("implement caching for the parser", FIX),
    ("rename the helper functions", FIX),
    ("open a pull request for the retry change", FIX),
    ("Propose a small PR that adds type hints", FIX),
    ("submit PRs for every typo", FIX),
    ("fix the bug in issue 42", FIX_GITHUB),
    ("refactor the code mentioned in this github issue", FIX_GITHUB),
    ("suggest how to simplify redirects", (R, A, G)),
    ("outline a better retry strategy", (R, A, G)),
    ("plan the retry logic", (R, A, G)),
    ("explain how sessions work", (R, A)),
    ("explain error handling in sessions", (R, A)),
    ("review the authentication module", (R, A)),
    ("summarize the cookie code", (R, A)),
    ("review the open issues about cookies", (R, H, A)),
    ("list open issues", (H, A)),
    ("what is on github", (H, A)),
    ("how does the HTTPAdapter send requests?", (R, A)),
    ("what does the config module do?", (R, A)),
    ("hello", (R, A)),
]


@pytest.mark.parametrize("goal,kinds", CASES)
def test_the_plan_fits_the_goal(goal: str, kinds: tuple) -> None:
    assert RuleBasedPlanner().plan(goal).kinds() == kinds


def test_plans_are_between_two_and_five_steps_and_all_sizes_are_used() -> None:
    sizes = {len(RuleBasedPlanner().plan(g).steps) for g, _ in CASES}
    assert sizes == {2, 3, 4, 5}


def test_the_intended_refactor_plan_from_the_roadmap() -> None:
    assert RuleBasedPlanner().plan("refactor request handling").kinds() == (R, A, G, V)


# --- every step can really run ----------------------------------------------------


@pytest.mark.parametrize("goal", [g for g, _ in CASES] + [
    "!!!", "pr", "pull request", "fix", "x" * 500, "Fix the bug AND open a PR for issue 3",
])
def test_every_step_in_every_rule_plan_is_one_execute_can_run(goal: str) -> None:
    plan = RuleBasedPlanner().plan(goal)
    assert 2 <= plan.step_count <= 5
    assert set(plan.kinds()) <= RUNNABLE
    assert [s.id for s in plan.steps] == list(range(1, plan.step_count + 1))


def test_an_empty_goal_is_still_rejected_clearly() -> None:
    with pytest.raises(ValueError):
        RuleBasedPlanner().plan("   ")


# --- word matching ---------------------------------------------------------------------


@pytest.mark.parametrize("goal", ["FIX THE BUG", "fix the bug!!!", "(fix) the bug", "fix:the-bug", "Fix\tthe\nbug"])
def test_capitals_and_punctuation_do_not_matter(goal: str) -> None:
    assert RuleBasedPlanner().plan(goal).kinds() == FIX


@pytest.mark.parametrize("word", ["bugs", "crashes", "crashing", "failed", "failing", "errors", "broken", "regression"])
def test_common_word_forms_count_as_a_problem(word: str) -> None:
    assert RuleBasedPlanner().plan(f"the parser has {word}").kinds() == FIX


@pytest.mark.parametrize("goal", [
    "refactor request handling", "optimize request retries", "explain how a request flows",
    "the request object", "request",
])
def test_the_word_request_alone_never_makes_a_pull_request_plan(goal: str) -> None:
    plan = RuleBasedPlanner().plan(goal)
    assert "pull request" not in " ".join(s.description.lower() for s in plan.steps)
    assert not any("pull request" in s.description.lower() for s in plan.steps)


@pytest.mark.parametrize("goal", ["open a PR", "make PRs", "send a pull request", "send Pull  Requests", "pull request!"])
def test_pull_request_goals_get_the_pull_request_plan(goal: str) -> None:
    plan = RuleBasedPlanner().plan(goal)
    assert plan.kinds() == FIX
    assert "pull request" in plan.steps[2].description.lower()


@pytest.mark.parametrize("goal", ["pull the latest request", "request a pull of the data"])
def test_pull_and_request_apart_are_not_a_pull_request(goal: str) -> None:
    plan = RuleBasedPlanner().plan(goal)
    assert not any("pull request" in s.description.lower() for s in plan.steps)


def test_backoff_is_no_longer_a_keyword() -> None:
    assert RuleBasedPlanner().plan("backoff").kinds() == (R, A)
    assert RuleBasedPlanner().plan("explain backoff").kinds() == (R, A)


def test_repo_and_repository_do_not_trigger_the_issues_step() -> None:
    assert RuleBasedPlanner().plan("describe the repository").kinds() == (R, A)
    assert RuleBasedPlanner().plan("how is the repo laid out").kinds() == (R, A)


# --- which intent wins ----------------------------------------------------------------------


def test_when_intents_mix_the_stronger_one_wins() -> None:
    plan = RuleBasedPlanner().plan
    assert plan("fix the bug and open a PR").kinds() == FIX
    assert "fix" in plan("fix the bug and open a PR").steps[2].description.lower()
    assert "pull request" in plan("open a PR to refactor the cache").steps[2].description.lower()
    assert plan("refactor and suggest names").kinds() == FIX          # change beats suggest
    assert plan("suggest a fix for the crash").kinds() == FIX         # fix beats suggest
    assert plan("explain why it crashes").kinds() == (R, A)           # asked to explain: review
    assert plan("why does login crash").kinds() == FIX                # a symptom without "explain"
    assert plan("review the fix for the login bug").kinds() == FIX    # fix verb wins


# --- honest texts -------------------------------------------------------------------------------


def _all_plans():
    return [RuleBasedPlanner().plan(g) for g, _ in CASES]


def test_validation_is_never_described_as_running_tests_or_lint() -> None:
    for plan in _all_plans():
        for step in plan.steps:
            text = step.description.lower()
            if step.kind == V:
                assert "syntax and import checks" in text and "not run" in text
            assert "test suite" not in text and "run tests" not in text and "tests + lint" not in text


def test_generate_says_the_diff_is_not_applied_or_not_validated() -> None:
    for plan in _all_plans():
        for step in plan.steps:
            if step.kind == G:
                text = step.description.lower()
                assert "diff" in text and ("not applied" in text or "not validated" in text or "separate step" in text)


def test_the_issues_step_says_what_it_needs() -> None:
    plan = RuleBasedPlanner().plan("fix the bug in issue 42")
    assert "open issues" in plan.steps[1].description and "owner and repo" in plan.steps[1].description


# --- the AI planner ------------------------------------------------------------------------------


class _LLM:
    model = "static"

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.text


def test_the_ai_planner_prompt_lists_only_kinds_that_can_run() -> None:
    for kind in ("retrieve", "analyze", "github_read", "generate", "validate"):
        assert kind in PLANNER_SYSTEM_PROMPT
    assert "human_approval" not in PLANNER_SYSTEM_PROMPT and "draft_pr" not in PLANNER_SYSTEM_PROMPT
    assert "does not run tests or lint" in PLANNER_SYSTEM_PROMPT


def test_the_ai_planner_still_decides_its_own_number_of_steps() -> None:
    reply = json.dumps({"steps": [
        {"kind": "retrieve", "description": "a"}, {"kind": "generate", "description": "b"},
        {"kind": "validate", "description": "c"},
    ]})
    plan = LLMPlanner(llm=_LLM(reply)).plan("anything")
    assert plan.strategy == "llm" and plan.kinds() == (R, G, V)


# --- wiring -------------------------------------------------------------------------------------


def test_get_planner_no_longer_takes_an_unused_parameter() -> None:
    assert list(inspect.signature(get_planner).parameters) == []
    assert isinstance(get_planner(), RuleBasedPlanner)


class _Issues:
    def __init__(self) -> None:
        self.coords: list = []

    def list_issues(self, coord, state="open", per_page=5):
        self.coords.append(coord.slug())
        return [SimpleNamespace(number=7, title="Retries crash", author="sam")]


def _ctx(prior: dict | None = None) -> StepContext:
    return StepContext(goal="g", step=PlanStep(id=2, kind=H, description="d"), prior_outputs=prior or {})


def test_the_issues_step_reads_the_repo_named_in_the_request() -> None:
    api = _Issues()
    out = GitHubReadExecutor(api).run(_ctx({REPO_TARGET_KEY: RepoTarget(owner="psf", repo="requests")}))
    assert api.coords == ["psf/requests"] and "#7  Retries crash" in out


def test_the_request_repo_wins_over_a_repo_set_on_the_server() -> None:
    from backend.github.models import RepoCoord

    api = _Issues()
    ex = GitHubReadExecutor(api, coord=RepoCoord(owner="server", repo="default"))
    ex.run(_ctx({REPO_TARGET_KEY: RepoTarget(owner="psf", repo="requests")}))
    ex.run(_ctx())
    assert api.coords == ["psf/requests", "server/default"]


def test_the_issues_step_is_skipped_with_a_clear_reason_when_no_repo_is_known() -> None:
    with pytest.raises(StepSkipped, match="Send owner and repo"):
        GitHubReadExecutor(_Issues()).run(_ctx())


# --- end to end through /execute ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear() -> Iterator:
    yield
    app.dependency_overrides.clear()


def test_refactor_request_handling_runs_every_step_with_none_skipped(tmp_path: Path) -> None:
    _, client = _execute_client(tmp_path, GOOD)
    body = client.post("/execute", json={
        "goal": "refactor request handling", "owner": "o", "repo": "r",
    }).json()
    assert [s["kind"] for s in body["steps"]] == ["retrieve", "analyze", "generate", "validate"]
    assert (body["completed"], body["skipped"], body["failed"], body["aborted"]) == (4, 0, 0, False)


def test_a_fix_goal_also_runs_every_step_with_none_skipped(tmp_path: Path) -> None:
    _, client = _execute_client(tmp_path, GOOD)
    body = client.post("/execute", json={"goal": "fix the add bug", "owner": "o", "repo": "r"}).json()
    assert (body["completed"], body["skipped"], body["failed"]) == (4, 0, 0)


def test_the_issues_step_runs_through_execute_when_a_repo_is_sent(tmp_path: Path) -> None:
    from backend.agent.engine import ExecutionEngine
    from backend.agent.executor import make_default_executors
    from backend.agent.fix_generator import FixGenerator
    from backend.agent.validation_checks import SyntaxCheck
    from backend.agent.validation_pipeline import ValidationPipeline
    from backend.main import get_execution_engine, get_llm_client, get_rag_pipeline, get_vector_store
    from backend.rag.llm import EchoLLMClient
    from backend.rag.pipeline import RAGPipeline
    from tests.test_index_safety import _store
    from tests.test_validate_repo_routes import FakeProvider, CALC, _StaticLLM

    store = _store(tmp_path / "v")
    rag = RAGPipeline(vector_store=store, llm=EchoLLMClient())
    api = _Issues()
    validator = ValidationPipeline(
        checks=(SyntaxCheck(),), sandbox_parent=tmp_path / "boxes",
        snapshot_provider=FakeProvider({"pkg/calc.py": CALC}),
    )
    executors = make_default_executors(
        store, rag, github=api, generator=FixGenerator(llm=_StaticLLM(GOOD)),
        validation_pipeline=validator,
    )
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_rag_pipeline] = lambda: rag
    app.dependency_overrides[get_llm_client] = lambda: EchoLLMClient()
    app.dependency_overrides[get_execution_engine] = lambda: ExecutionEngine(executors=executors)
    body = TestClient(app).post("/execute", json={
        "goal": "fix the add bug from issue 3", "owner": "psf", "repo": "requests",
    }).json()
    assert [s["kind"] for s in body["steps"]] == ["retrieve", "github_read", "analyze", "generate", "validate"]
    assert api.coords == ["psf/requests"]
    github_step = body["steps"][1]
    assert github_step["status"] == "completed" and "Retries crash" in github_step["output"]


def test_the_issues_step_is_skipped_honestly_through_execute_without_a_repo(tmp_path: Path) -> None:
    _, client = _execute_client(tmp_path, GOOD)
    body = client.post("/execute", json={"goal": "fix the add bug from issue 3"}).json()
    github_step = next(s for s in body["steps"] if s["kind"] == "github_read")
    assert github_step["status"] == "skipped" and github_step["error"]


# --- the plan viewer script ----------------------------------------------------------------------


def _script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "show_plan.py"
    spec = importlib.util.spec_from_file_location("show_plan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_show_plan_prints_the_plan_for_a_goal(capsys) -> None:
    assert _script().main(["show_plan", "refactor request handling"]) == 0
    out = capsys.readouterr().out
    assert "Goal: refactor request handling   (4 steps)" in out
    assert "[retrieve]" in out and "[validate]" in out and "[draft_pr]" not in out


def test_show_plan_without_a_goal_shows_the_samples(capsys) -> None:
    module = _script()
    assert module.main(["show_plan"]) == 0
    assert capsys.readouterr().out.count("Goal: ") == len(module.SAMPLES)
