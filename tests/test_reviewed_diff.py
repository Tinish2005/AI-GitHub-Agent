"""Loop 4 tests: the reviewed diff is the diff that is validated and submitted (OFFLINE)."""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.agent.draft_pr_service import FakeDraftPRService
from backend.agent.executor import PATCH_MARKER, make_default_executors
from backend.agent.engine import ExecutionEngine
from backend.agent.fix_generator import FixGenerator, proposal_from_reviewed
from backend.agent.fix_models import (
    MAX_DIFF_CHARS,
    FixProposal,
    ReviewedDiffError,
    ReviewedProposal,
    diff_fingerprint,
)
from backend.agent.validation_checks import SyntaxCheck
from backend.agent.validation_pipeline import ValidationPipeline
from backend.config import Settings, get_settings
from backend.indexing.ast_parser import parse_source
from backend.main import (
    PR_NEEDS_REVIEWED_MESSAGE,
    REGENERATED_WARNING,
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
from tests.test_index_safety import _store
from tests.test_validation_real_repo import CALC, CALC_FIXED, COMMIT, FakeProvider

GOAL = "fix the add bug"
REPO = {"owner": "o", "repo": "r"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_diff(path: str, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    )).strip()


class FlakyLLM:
    """Gives a DIFFERENT, valid diff every time it is asked, like a real AI can."""

    model = "flaky"

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        self.prompts.append(user_prompt)
        new = CALC_FIXED + f"\n# attempt {self.calls}\n"
        return json.dumps({
            "explanation": f"attempt {self.calls}",
            "diff": make_diff("pkg/calc.py", CALC, new),
            "confidence": 0.8,
        })


class Env(SimpleNamespace):
    def propose(self) -> dict:
        resp = self.client.post("/fix/propose", json={"goal": GOAL, **REPO})
        assert resp.status_code == 200, resp.text
        return resp.json()

    @staticmethod
    def reviewed(proposed: dict, **changes) -> dict:
        data = {
            "goal": GOAL,
            "explanation": proposed["explanation"],
            "diff": proposed["diff"],
            "confidence": proposed["confidence"],
            "diff_sha256": proposed["diff_sha256"],
        }
        data.update(changes)
        return data


def build_env(tmp_path: Path, *, allow_regenerated: bool = False) -> Env:
    store = _store(tmp_path / "vdb")
    store.add_chunks(parse_source(CALC, Path("pkg/calc.py")), extra_metadata={"repo": "o/r"})
    llm = FlakyLLM()
    provider = FakeProvider({"pkg/calc.py": CALC})
    validator = ValidationPipeline(
        checks=(SyntaxCheck(),), sandbox_parent=tmp_path / "boxes", snapshot_provider=provider,
    )
    pr_service = FakeDraftPRService(next_number=9)
    rag = RAGPipeline(vector_store=store, llm=EchoLLMClient())
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_rag_pipeline] = lambda: rag
    app.dependency_overrides[get_llm_client] = lambda: EchoLLMClient()
    app.dependency_overrides[get_fix_generator] = lambda: FixGenerator(llm=llm)
    app.dependency_overrides[get_validation_pipeline] = lambda: validator
    app.dependency_overrides[get_draft_pr_service] = lambda: pr_service
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, allow_regenerated_pr=allow_regenerated,
    )
    return Env(client=TestClient(app), llm=llm, provider=provider, pr=pr_service, store=store,
               validator=validator, tmp=tmp_path)


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    yield build_env(tmp_path)
    app.dependency_overrides.clear()


@pytest.fixture
def hatch_env(tmp_path: Path) -> Iterator[Env]:
    yield build_env(tmp_path, allow_regenerated=True)
    app.dependency_overrides.clear()


def post_validate(env: Env, proposal: dict | None, **extra):
    body = {"goal": GOAL, **REPO, **extra}
    if proposal is not None:
        body["proposal"] = proposal
    return env.client.post("/fix/validate", json=body)


def post_pr(env: Env, proposal: dict | None, **extra):
    body = {"goal": GOAL, **REPO, **extra}
    if proposal is not None:
        body["proposal"] = proposal
    return env.client.post("/fix/pr", json=body)


# --- the regression the loop exists for ----------------------------------------------------


def test_the_displayed_validated_and_submitted_diffs_are_identical(env: Env) -> None:
    proposed = env.propose()
    reviewed = env.reviewed(proposed)

    validated = post_validate(env, reviewed).json()
    pr = post_pr(env, reviewed).json()

    submitted = env.pr.calls[-1].proposal_diff
    assert env.llm.calls == 1, "the server must not ask the AI again after the review"
    assert submitted == proposed["diff"]
    assert proposed["diff_sha256"] == sha(proposed["diff"])
    assert validated["diff_sha256"] == pr["diff_sha256"] == sha(submitted) == proposed["diff_sha256"]
    assert validated["diff_source"] == pr["diff_source"] == "reviewed"
    assert validated["warning"] == "" and pr["warning"] == ""
    assert validated["passed"] is True and pr["created"] is True
    assert env.pr.calls[-1].validation_passed is True


def test_the_ai_stand_in_really_gives_a_different_diff_each_time(env: Env) -> None:
    first, second = env.propose(), env.propose()
    assert first["diff"] != second["diff"]
    assert first["diff_sha256"] != second["diff_sha256"]


def test_without_a_reviewed_proposal_the_old_problem_is_visible(hatch_env: Env) -> None:
    proposed = hatch_env.propose()
    validated = post_validate(hatch_env, None).json()
    assert validated["diff_sha256"] != proposed["diff_sha256"]
    assert validated["diff_source"] == "regenerated" and validated["warning"] == REGENERATED_WARNING


def test_the_full_known_bug_scenario_ends_with_the_same_diff_in_the_pr(env: Env) -> None:
    # 1. question, answered from the right repo and the right file
    qa = env.client.post("/qa", json={"question": "how does add work", **REPO}).json()
    # Windows gives backslashes in file paths, Linux gives slashes: compare them the same way.
    assert {s["file_path"].replace("\\", "/") for s in qa["sources"]} == {"pkg/calc.py"}
    # 2. relevant retrieval reaches the fix prompt, and a fix is proposed
    proposed = env.propose()
    assert "def add" in env.llm.prompts[0] and "return a - b" in env.llm.prompts[0]
    # 3. validated against the real repository files
    reviewed = env.reviewed(proposed)
    validated = post_validate(env, reviewed, ref="main").json()
    assert validated["passed"] is True and validated["base_commit"] == COMMIT
    assert env.provider.calls[-1]["owner"] == "o" and env.provider.calls[-1]["ref"] == "main"
    # 4. the same diff goes through PR creation
    pr = post_pr(env, reviewed, base_branch="main").json()
    request = env.pr.calls[-1]
    assert pr["created"] is True and request.base_branch == "main"
    assert request.proposal_diff == proposed["diff"]
    assert request.proposal_explanation == proposed["explanation"]
    assert sha(request.proposal_diff) == validated["diff_sha256"] == pr["diff_sha256"]
    assert env.llm.calls == 1


# --- /fix/pr needs the reviewed proposal ---------------------------------------------------------


def test_a_pull_request_without_a_reviewed_proposal_is_refused(env: Env) -> None:
    resp = post_pr(env, None)
    assert resp.status_code == 422 and resp.json()["detail"] == PR_NEEDS_REVIEWED_MESSAGE
    assert env.llm.calls == 0 and env.pr.calls == []


def test_the_escape_hatch_allows_a_regenerated_diff_and_says_so(hatch_env: Env) -> None:
    resp = post_pr(hatch_env, None)
    body = resp.json()
    assert resp.status_code == 200 and body["created"] is True
    assert body["diff_source"] == "regenerated" and body["warning"] == REGENERATED_WARNING
    assert body["diff_sha256"] == sha(hatch_env.pr.calls[-1].proposal_diff)
    assert hatch_env.llm.calls == 1


def test_a_reviewed_proposal_does_not_touch_the_ai_or_the_search(env: Env) -> None:
    proposed = env.propose()
    calls_after_propose = env.llm.calls

    class _NoSearch:
        def query(self, *args, **kwargs):
            raise AssertionError("the reviewed flow must not search")

    app.dependency_overrides[get_vector_store] = lambda: _NoSearch()
    reviewed = env.reviewed(proposed)
    assert post_validate(env, reviewed).status_code == 200
    assert post_pr(env, reviewed).status_code == 200
    assert env.llm.calls == calls_after_propose


# --- a changed diff is caught --------------------------------------------------------------------


@pytest.mark.parametrize("route", [post_validate, post_pr])
def test_a_diff_that_no_longer_matches_its_fingerprint_is_refused(env: Env, route) -> None:
    proposed = env.propose()
    tampered = env.reviewed(proposed, diff=proposed["diff"].replace("a + b", "a * b"))
    resp = route(env, tampered)
    assert resp.status_code == 400
    assert "changed after it was reviewed" in resp.json()["detail"]
    assert env.pr.calls == []


def test_a_different_diff_sent_without_a_fingerprint_is_used_as_sent_never_swapped(env: Env) -> None:
    other = make_diff("pkg/calc.py", CALC, CALC_FIXED.replace("def sub", "def subtract"))
    reviewed = {"goal": GOAL, "explanation": "mine", "diff": other}
    validated = post_validate(env, reviewed).json()
    pr = post_pr(env, reviewed).json()
    assert validated["diff_sha256"] == pr["diff_sha256"] == sha(other)
    assert env.pr.calls[-1].proposal_diff == other
    assert env.llm.calls == 0


def test_the_fingerprint_check_ignores_capitals_and_spaces(env: Env) -> None:
    proposed = env.propose()
    reviewed = env.reviewed(proposed, diff_sha256="  " + proposed["diff_sha256"].upper() + " ")
    assert post_validate(env, reviewed).status_code == 200


# --- the server checks the diff itself ------------------------------------------------------------


def test_a_broken_diff_is_not_valid_whatever_the_client_claims(env: Env) -> None:
    reviewed = {"goal": GOAL, "explanation": "e", "diff": "this is not a diff",
                "is_valid": True, "hunks": [], "validation_error": ""}
    validated = post_validate(env, reviewed).json()
    assert validated["passed"] is False and "Proposal is not valid" in validated["error"]
    assert validated["diff_sha256"] == sha("this is not a diff")
    pr = post_pr(env, reviewed).json()
    assert pr["created"] is False and pr["diff_sha256"] == sha("this is not a diff")


def test_a_diff_that_does_not_fit_the_real_repo_is_not_validated_or_submitted(env: Env) -> None:
    wrong = make_diff("pkg/calc.py", CALC.replace("a - b", "a * b"), CALC)
    reviewed = {"goal": GOAL, "explanation": "e", "diff": wrong}
    validated = post_validate(env, reviewed).json()
    assert validated["passed"] is False and "pkg/calc.py" in validated["error"]
    assert post_pr(env, reviewed).json()["created"] is False


@pytest.mark.parametrize("proposal", [
    {"goal": GOAL, "explanation": "e", "diff": ""},
    {"goal": GOAL, "explanation": "", "diff": "x"},
    {"goal": "", "explanation": "e", "diff": "x"},
    {"explanation": "e", "diff": "x"},
    {"goal": GOAL, "explanation": "e", "diff": "x" * (MAX_DIFF_CHARS + 1)},
    {"goal": GOAL, "explanation": "e", "diff": "x", "confidence": 2},
])
def test_incomplete_or_oversized_proposals_are_refused(env: Env, proposal: dict) -> None:
    assert post_validate(env, proposal).status_code == 422
    assert post_pr(env, proposal).status_code == 422
    assert env.pr.calls == []


def test_a_proposal_at_the_size_limit_is_accepted_by_the_model() -> None:
    assert ReviewedProposal(goal="g", explanation="e", diff="x" * MAX_DIFF_CHARS).diff


def test_reviewed_proposal_defaults() -> None:
    p = ReviewedProposal(goal="g", explanation="e", diff="d")
    assert (p.confidence, p.model, p.diff_sha256) == (0.5, "reviewed", None)


# --- old calls still work for validate ---------------------------------------------------------------


def test_validate_with_only_a_goal_still_works_and_is_labelled(env: Env) -> None:
    body = post_validate(env, None).json()
    assert body["passed"] is True
    assert body["diff_source"] == "regenerated" and body["warning"] == REGENERATED_WARNING
    assert body["diff_sha256"] and env.llm.calls == 1


# --- the fingerprint itself ----------------------------------------------------------------------------


def test_the_fingerprint_is_the_sha256_of_the_exact_text() -> None:
    assert diff_fingerprint("abc") == hashlib.sha256(b"abc").hexdigest()
    assert diff_fingerprint("é\n") == hashlib.sha256("é\n".encode("utf-8")).hexdigest()
    assert diff_fingerprint("a") != diff_fingerprint("a\n")


def test_every_proposal_carries_its_fingerprint() -> None:
    p = FixProposal(goal="g", explanation="e", diff="d", model="m", confidence=0.5)
    assert p.diff_sha256 == sha("d")
    assert p.model_dump()["diff_sha256"] == sha("d")
    assert json.loads(p.model_dump_json())["diff_sha256"] == sha("d")


def test_the_builder_refuses_a_wrong_fingerprint_and_keeps_the_diff_exactly() -> None:
    diff = make_diff("pkg/calc.py", CALC, CALC_FIXED)
    good = proposal_from_reviewed(ReviewedProposal(goal="g", explanation="e", diff=diff, diff_sha256=sha(diff)))
    assert good.diff == diff and good.is_valid is True and good.files_changed == 1
    with pytest.raises(ReviewedDiffError):
        proposal_from_reviewed(ReviewedProposal(goal="g", explanation="e", diff=diff, diff_sha256="0" * 64))


def test_validation_results_carry_the_fingerprint_on_every_path(tmp_path: Path) -> None:
    pipe = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path / "b")
    new_file = make_diff("a.py", "", "x = 1\n").replace("--- a/a.py", "--- /dev/null")
    ok = pipe.validate(FixProposal(goal="g", explanation="e", diff=new_file, model="m", confidence=1, is_valid=True))
    assert ok.passed and ok.diff_sha256 == sha(new_file)
    invalid = pipe.validate(FixProposal(goal="g", explanation="e", diff="zz", model="m", confidence=1, is_valid=False))
    assert invalid.diff_sha256 == sha("zz")
    escaping = "--- a/x.py\n+++ b/../../x.py\n@@ -1 +1 @@\n-a\n+b\n"
    bad_patch = pipe.validate(FixProposal(
        goal="g", explanation="e", diff=escaping, model="m", confidence=1, is_valid=True,
    ))
    assert bad_patch.passed is False and bad_patch.diff_sha256 == sha(escaping)


# --- the escape hatch setting ----------------------------------------------------------------------------


def test_the_escape_hatch_defaults_to_off_and_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALLOW_REGENERATED_PR", raising=False)
    assert Settings(_env_file=None).allow_regenerated_pr is False
    monkeypatch.setenv("ALLOW_REGENERATED_PR", "true")
    assert Settings(_env_file=None).allow_regenerated_pr is True


# --- /execute shows the same fingerprint at both steps ----------------------------------------------------------


def test_execute_shows_the_same_fingerprint_for_the_generated_and_the_checked_diff(env: Env) -> None:
    executors = make_default_executors(
        env.store, RAGPipeline(vector_store=env.store, llm=EchoLLMClient()),
        generator=FixGenerator(llm=env.llm), validation_pipeline=env.validator,
    )
    app.dependency_overrides[get_execution_engine] = lambda: ExecutionEngine(executors=executors)
    body = env.client.post("/execute", json={"goal": GOAL, **REPO}).json()
    steps = {s["kind"]: s for s in body["steps"]}
    generated, checked = steps["generate"]["output"], steps["validate"]["output"]
    diff = generated.split(PATCH_MARKER + "\n", 1)[1]
    assert f"Diff SHA-256: {sha(diff)}" in generated
    assert f"Checked diff SHA-256: {sha(diff)}" in checked
    assert (body["completed"], body["skipped"], body["failed"]) == (4, 0, 0)
