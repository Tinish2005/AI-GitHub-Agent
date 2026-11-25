"""Loop 4b tests: the real PR service handles new, changed and deleted files (OFFLINE).

GitHub is simulated by a small stateful fake that follows the rules the real
API enforces (an update needs the file's blob sha, a new file must not exist,
and so on). These tests prove our logic. They do NOT prove the real GitHub API
behaves the same; scripts/live_check_pr.py does that on a repo you own.
"""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote

import httpx
import pytest
from fastapi.testclient import TestClient
from unidiff import PatchSet

from backend.agent.draft_pr_models import DraftPRRequest
from backend.agent.draft_pr_service import GitHubDraftPRService
from backend.agent.patching import apply_patch_to_text, dominant_eol
from backend.agent.validation_pipeline import PatchApplyError
from backend.main import app, get_draft_pr_service, get_fix_generator, get_validation_pipeline, get_settings
from backend.config import Settings

TOKEN = "ghp_super_secret_token_value"
BASE_SHA = "b" * 40

CALC = "import os\n\ndef add(a, b):\n    return a - b\n\ndef sub(a, b):\n    return a - b\n"
CALC_FIXED = CALC.replace("def add(a, b):\n    return a - b", "def add(a, b):\n    return a + b")
UTIL = "def helper():\n    return 1\n"


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def make_diff(path: str, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    )).rstrip("\n")


def new_file_diff(path: str, text: str) -> str:
    return "".join(difflib.unified_diff(
        [], text.splitlines(keepends=True), fromfile="/dev/null", tofile=f"b/{path}",
    )).rstrip("\n")


def delete_diff(path: str, text: str) -> str:
    return "".join(difflib.unified_diff(
        text.splitlines(keepends=True), [], fromfile=f"a/{path}", tofile="/dev/null",
    )).rstrip("\n")


class FakeGitHub:
    """A tiny GitHub: one repo, a base commit, branches, files, and the PR endpoint."""

    def __init__(self, files: dict) -> None:
        self.base_files = {p: (t if isinstance(t, bytes) else t.encode("utf-8")) for p, t in files.items()}
        self.branches = {"main": BASE_SHA}
        self.branch_files: dict = {}
        self.special: dict = {}   # path -> json to return for the contents GET
        self.fail: dict = {}      # (METHOD, substring) -> status code
        self.log: list = []       # (METHOD, path)
        self.requests: list = []  # full httpx requests
        self.prs: list = []
        self.commits: list = []
        self.pr_state = "open"

    # ---- helpers for tests ----
    def calls(self, method: str, fragment: str = "") -> list:
        return [p for m, p in self.log if m == method and fragment in p]

    def text_on(self, branch: str, path: str) -> str | None:
        files = self.branch_files.get(branch)
        return None if files is None or path not in files else files[path].decode("utf-8")

    # ---- the fake server ----
    def handler(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        self.log.append((method, path))
        self.requests.append(request)
        for (m, fragment), status in self.fail.items():
            if m == method and fragment in path:
                return httpx.Response(status, json={"message": "injected failure"})
        body = json.loads(request.content) if request.content else {}
        repo = "/repos/o/r"

        if method == "GET" and path.startswith(f"{repo}/git/ref/heads/"):
            branch = unquote(path[len(f"{repo}/git/ref/heads/"):])
            if branch in self.branches:
                return httpx.Response(200, json={"object": {"sha": self.branches[branch]}})
            return httpx.Response(404, json={"message": "Not Found"})

        if method == "GET" and path.startswith(f"{repo}/contents/"):
            name = unquote(path[len(f"{repo}/contents/"):])
            ref = request.url.params.get("ref")
            if ref in self.branch_files:  # a read from a branch we created
                files = self.branch_files[ref]
                if name not in files:
                    return httpx.Response(404, json={"message": "Not Found"})
                data = files[name]
                return httpx.Response(200, json={
                    "type": "file", "encoding": "base64", "content": base64.b64encode(data).decode("ascii"),
                    "sha": blob_sha(data), "size": len(data), "path": name,
                })
            if ref != BASE_SHA:
                return httpx.Response(404, json={"message": "No commit found for the ref"})
            if name in self.special:
                return httpx.Response(200, json=self.special[name])
            if name not in self.base_files:
                return httpx.Response(404, json={"message": "Not Found"})
            data = self.base_files[name]
            wrapped = "\n".join(re.findall(".{1,60}", base64.b64encode(data).decode("ascii")))
            return httpx.Response(200, json={
                "type": "file", "encoding": "base64", "content": wrapped,
                "sha": blob_sha(data), "size": len(data), "path": name,
            })

        if method == "POST" and path == f"{repo}/git/refs":
            if body.get("sha") != BASE_SHA:
                return httpx.Response(422, json={"message": "Object does not exist"})
            name = body["ref"].removeprefix("refs/heads/")
            self.branches[name] = body["sha"]
            self.branch_files[name] = dict(self.base_files)
            return httpx.Response(201, json={"ref": body["ref"]})

        if method in ("PUT", "DELETE") and path.startswith(f"{repo}/contents/"):
            name = unquote(path[len(f"{repo}/contents/"):])
            files = self.branch_files.get(body.get("branch"))
            if files is None:
                return httpx.Response(404, json={"message": "Branch not found"})
            exists = name in files
            if method == "PUT":
                if exists and body.get("sha") != blob_sha(files[name]):
                    return httpx.Response(409, json={"message": "sha does not match"})
                if not exists and body.get("sha"):
                    return httpx.Response(422, json={"message": "sha supplied for a new file"})
                files[name] = base64.b64decode(body["content"])
            else:
                if not exists or body.get("sha") != blob_sha(files[name]):
                    return httpx.Response(409, json={"message": "sha does not match"})
                del files[name]
            self.commits.append((method, name, body.get("message")))
            return httpx.Response(200, json={"commit": {"sha": "c" * 40}})

        if method == "POST" and path == f"{repo}/pulls":
            if body.get("draft") is not True:
                return httpx.Response(422, json={"message": "must be a draft"})
            self.prs.append(body)
            return httpx.Response(201, json={"html_url": "https://github.com/o/r/pull/5", "number": 5})

        if method == "GET" and path == f"{repo}/pulls/5":
            return httpx.Response(200, json={"draft": bool(self.prs and self.prs[-1].get("draft")), "state": self.pr_state})
        if method == "PATCH" and path == f"{repo}/pulls/5":
            self.pr_state = body.get("state", self.pr_state)
            return httpx.Response(200, json={"state": self.pr_state})

        if method == "DELETE" and path.startswith(f"{repo}/git/refs/heads/"):
            branch = unquote(path[len(f"{repo}/git/refs/heads/"):])
            self.branches.pop(branch, None)
            self.branch_files.pop(branch, None)
            return httpx.Response(204)

        return httpx.Response(404, json={"message": f"unhandled {method} {path}"})


def make_service(gh: FakeGitHub) -> GitHubDraftPRService:
    return GitHubDraftPRService(token=TOKEN, transport=httpx.MockTransport(gh.handler))


def request_for(diff: str, **overrides) -> DraftPRRequest:
    data = dict(
        owner="o", repo="r", base_branch="main", goal="fix the add bug",
        proposal_explanation="swap - for +", proposal_diff=diff,
        confidence=0.9, validation_passed=True, validation_score=1.0,
    )
    data.update(overrides)
    return DraftPRRequest(**data)


def nothing_was_created(gh: FakeGitHub) -> bool:
    return not (gh.calls("POST") or gh.calls("PUT") or gh.calls("DELETE")) and not gh.prs


# --- changing an existing file -----------------------------------------------------------


def test_a_change_to_an_existing_file_becomes_a_draft_pr_with_exactly_the_reviewed_change() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    diff = make_diff("pkg/calc.py", CALC, CALC_FIXED)
    result = make_service(gh).create(request_for(diff))

    assert result.created is True and result.pr_number == 5 and result.error == ""
    committed = gh.text_on(result.branch, "pkg/calc.py")
    assert committed == CALC_FIXED
    # The strongest check: diffing the base against what was committed gives the reviewed diff back.
    assert make_diff("pkg/calc.py", CALC, committed) == diff
    assert gh.prs[0]["draft"] is True and gh.prs[0]["base"] == "main" and gh.prs[0]["head"] == result.branch
    assert diff in gh.prs[0]["body"]
    assert gh.base_files["pkg/calc.py"].decode() == CALC  # the base branch is untouched


def test_the_update_carries_the_blob_sha_of_the_file_it_replaces() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    make_service(gh).create(request_for(make_diff("pkg/calc.py", CALC, CALC_FIXED)))
    put = next(r for r in gh.requests if r.method == "PUT")
    body = json.loads(put.content)
    assert body["sha"] == blob_sha(CALC.encode()) and body["branch"].startswith("agent/")


def test_files_are_read_from_the_base_commit_and_the_branch_starts_from_that_commit() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    make_service(gh).create(request_for(make_diff("pkg/calc.py", CALC, CALC_FIXED)))
    reads = [r for r in gh.requests if r.method == "GET" and "/contents/" in r.url.path]
    assert reads and all(r.url.params["ref"] == BASE_SHA for r in reads)
    create_ref = next(r for r in gh.requests if r.method == "POST" and r.url.path.endswith("/git/refs"))
    assert json.loads(create_ref.content)["sha"] == BASE_SHA
    order = [m + " " + p.split("/")[-1] for m, p in gh.log]
    assert order.index("GET calc.py") < order.index("POST refs")


def test_a_diff_with_wrong_line_numbers_still_applies_when_the_lines_match() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    diff = make_diff("pkg/calc.py", CALC, CALC_FIXED).replace("@@ -1,", "@@ -40,", 1)
    result = make_service(gh).create(request_for(diff))
    assert result.created is True and gh.text_on(result.branch, "pkg/calc.py") == CALC_FIXED


def test_several_hunks_in_one_file() -> None:
    old = "".join(f"line {i}\n" for i in range(1, 41))
    new = old.replace("line 3\n", "line three\n").replace("line 35\n", "line thirty-five\n")
    gh = FakeGitHub({"big.txt": old})
    diff = make_diff("big.txt", old, new)
    assert sum(1 for line in diff.splitlines() if line.startswith("@@")) == 2  # two separate hunks
    result = make_service(gh).create(request_for(diff))
    assert result.created is True and gh.text_on(result.branch, "big.txt") == new


def test_windows_line_endings_are_kept() -> None:
    crlf = CALC.replace("\n", "\r\n")
    gh = FakeGitHub({"pkg/calc.py": crlf})
    result = make_service(gh).create(request_for(make_diff("pkg/calc.py", CALC, CALC_FIXED)))
    assert result.created is True
    assert gh.text_on(result.branch, "pkg/calc.py") == CALC_FIXED.replace("\n", "\r\n")


def test_a_file_without_a_final_newline_can_be_extended() -> None:
    old = "a = 1\nb = 2"  # no newline at the end
    gh = FakeGitHub({"x.py": old})
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,3 @@\n a = 1\n b = 2\n+c = 3"
    result = make_service(gh).create(request_for(diff))
    assert result.created is True and gh.text_on(result.branch, "x.py") == "a = 1\nb = 2\nc = 3\n"


# --- new files and deletions ------------------------------------------------------------------


def test_a_new_file_is_created_without_a_sha() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    result = make_service(gh).create(request_for(new_file_diff("pkg/util.py", UTIL)))
    assert result.created is True and gh.text_on(result.branch, "pkg/util.py") == UTIL
    put = next(r for r in gh.requests if r.method == "PUT")
    assert "sha" not in json.loads(put.content)


def test_a_deleted_file_is_removed_with_its_blob_sha() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC, "pkg/old.py": UTIL})
    result = make_service(gh).create(request_for(delete_diff("pkg/old.py", UTIL)))
    assert result.created is True
    assert gh.text_on(result.branch, "pkg/old.py") is None and gh.text_on(result.branch, "pkg/calc.py") == CALC
    delete = next(r for r in gh.requests if r.method == "DELETE" and "/contents/" in r.url.path)
    assert json.loads(delete.content)["sha"] == blob_sha(UTIL.encode())


def test_new_changed_and_deleted_files_in_one_pull_request() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC, "pkg/old.py": UTIL})
    diff = "\n".join([
        make_diff("pkg/calc.py", CALC, CALC_FIXED),
        new_file_diff("pkg/new.py", "x = 1\n"),
        delete_diff("pkg/old.py", UTIL),
    ])
    result = make_service(gh).create(request_for(diff))
    assert result.created is True
    assert [c[:2] for c in gh.commits] == [("PUT", "pkg/calc.py"), ("PUT", "pkg/new.py"), ("DELETE", "pkg/old.py")]
    assert gh.text_on(result.branch, "pkg/new.py") == "x = 1\n"


# --- a diff that does not fit leaves nothing behind -----------------------------------------------


def test_a_diff_that_does_not_fit_the_real_file_creates_nothing() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC.replace("a - b", "a * b")})
    result = make_service(gh).create(request_for(make_diff("pkg/calc.py", CALC, CALC_FIXED)))
    assert result.created is False
    assert "does not fit the repository" in result.error and "pkg/calc.py" in result.error
    assert nothing_was_created(gh)


@pytest.mark.parametrize("diff,message", [
    (make_diff("pkg/gone.py", CALC, CALC_FIXED), "does not exist in the repository"),
    (new_file_diff("pkg/calc.py", "x = 1\n"), "already exists"),
    (delete_diff("pkg/missing.py", UTIL), "does not exist in the repository"),
])
def test_wrong_assumptions_about_which_files_exist_are_caught_early(diff: str, message: str) -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    result = make_service(gh).create(request_for(diff))
    assert result.created is False and message in result.error
    assert nothing_was_created(gh)


def test_a_deletion_that_does_not_cover_the_whole_file_is_refused() -> None:
    gh = FakeGitHub({"pkg/old.py": "a\nb\nc\n"})
    diff = "--- a/pkg/old.py\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-a"
    result = make_service(gh).create(request_for(diff))
    assert result.created is False and "do not cover the whole file" in result.error
    assert nothing_was_created(gh)


# --- things the agent will never do ------------------------------------------------------------------


@pytest.mark.parametrize("diff,message", [
    (new_file_diff(".github/workflows/ci.yml", "on: push\n"), "workflow file"),
    (make_diff("../outside.py", "a\n", "b\n"), "Unsafe file path"),
    (new_file_diff("/abs/path.py", "x\n"), "Unsafe file path"),
    (new_file_diff(".git/config", "x\n"), "Unsafe file path"),
    ("--- a/pkg/calc.py\n+++ b/pkg/renamed.py\n@@ -1,1 +1,1 @@\n-import os\n+import sys", "renames are not supported"),
    ("this is not a diff", "no file changes"),
])
def test_unsafe_or_unsupported_diffs_are_refused_before_anything_is_created(diff: str, message: str) -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    result = make_service(gh).create(request_for(diff))
    assert result.created is False and message.lower() in result.error.lower()
    assert nothing_was_created(gh)


def test_a_diff_touching_too_many_files_is_refused() -> None:
    gh = FakeGitHub({})
    # Changes to existing files, so older unidiff versions can read it too.
    diff = "\n".join(make_diff(f"f{i}.py", "x = 1\n", "x = 2\n") for i in range(21))
    result = make_service(gh).create(request_for(diff))
    assert result.created is False and "limit is 20" in result.error
    assert nothing_was_created(gh)


def test_the_same_file_twice_in_one_diff_is_refused_whatever_the_unidiff_version() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    one = make_diff("pkg/calc.py", CALC, CALC_FIXED)
    for diff in (one + "\n" + one, one.replace("\n", "\n", 1) + "\n" + one + "\n"):
        result = make_service(gh).create(request_for(diff))
        assert result.created is False and "pkg/calc.py appears twice" in result.error
    assert nothing_was_created(gh)


def test_a_new_file_and_a_change_to_the_same_path_count_as_twice() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    diff = new_file_diff("pkg/calc.py", "x = 1\n") + "\n" + make_diff("pkg/calc.py", CALC, CALC_FIXED)
    result = make_service(gh).create(request_for(diff))
    assert result.created is False and "appears twice" in result.error


def test_changed_lines_that_look_like_file_headers_are_not_mistaken_for_files() -> None:
    from backend.agent.draft_pr_service import _first_duplicate_path

    sql_a = "-- note\nselect 1;\n"
    sql_b = "++ note\nselect 1;\n"
    one = make_diff("a.sql", sql_a, sql_b)
    two = make_diff("b.sql", sql_a, sql_b)
    assert "--- note" in one and "+++ note" in one
    assert _first_duplicate_path(one + "\n" + two) is None


def test_an_unreadable_diff_names_the_old_unidiff_as_a_likely_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata

    bad = "+++ b/x\n@@ -1 +1 @@\n-a\n+b"
    gh = FakeGitHub({})
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.7.5")
    old = make_service(gh).create(request_for(bad))
    assert "could not be read" in old.error and "unidiff>=1.0.0" in old.error and "0.7.5" in old.error
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.0.1")
    new = make_service(gh).create(request_for(bad))
    assert "could not be read" in new.error and "unidiff>=1.0.0" not in new.error
    assert nothing_was_created(gh)


@pytest.mark.parametrize("special,message", [
    ({"type": "symlink", "target": "x"}, "not a regular file"),
    ({"type": "file", "encoding": "base64", "content": "", "sha": "s", "size": 5_000_000}, "too large"),
    ({"type": "file", "encoding": "none", "content": "", "sha": "s", "size": 10}, "did not return the content"),
])
def test_special_files_are_refused(special: dict, message: str) -> None:
    gh = FakeGitHub({})
    gh.special["pkg/calc.py"] = special
    result = make_service(gh).create(request_for(make_diff("pkg/calc.py", CALC, CALC_FIXED)))
    assert result.created is False and message in result.error
    assert nothing_was_created(gh)


def test_a_directory_in_place_of_a_file_is_refused() -> None:
    gh = FakeGitHub({})
    gh.special["pkg/calc.py"] = [{"name": "x"}]
    result = make_service(gh).create(request_for(make_diff("pkg/calc.py", CALC, CALC_FIXED)))
    assert result.created is False and "not a regular file" in result.error


def test_a_file_that_is_not_utf8_text_is_refused() -> None:
    gh = FakeGitHub({"pkg/calc.py": b"\xff\xfe\x00binary"})
    result = make_service(gh).create(request_for(make_diff("pkg/calc.py", CALC, CALC_FIXED)))
    assert result.created is False and "not a UTF-8 text file" in result.error
    assert nothing_was_created(gh)


# --- GitHub failures --------------------------------------------------------------------------------------


def test_a_failure_part_way_removes_the_branch_and_opens_no_pr() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC, "pkg/util.py": UTIL})
    gh.fail[("PUT", "pkg/util.py")] = 500
    diff = make_diff("pkg/calc.py", CALC, CALC_FIXED) + "\n" + make_diff("pkg/util.py", UTIL, UTIL + "# x\n")
    result = make_service(gh).create(request_for(diff))
    assert result.created is False and "GitHub API error" in result.error
    assert "was removed" in result.error and result.branch
    assert gh.branches == {"main": BASE_SHA} and gh.prs == []


def test_if_the_branch_cannot_be_removed_the_message_says_so() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    gh.fail[("PUT", "pkg/calc.py")] = 500
    gh.fail[("DELETE", "/git/refs/heads/")] = 500
    result = make_service(gh).create(request_for(make_diff("pkg/calc.py", CALC, CALC_FIXED)))
    assert result.created is False and "could not be removed" in result.error and result.branch in result.error


def test_a_failed_pull_request_call_also_cleans_up_the_branch() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    gh.fail[("POST", "/pulls")] = 422
    result = make_service(gh).create(request_for(make_diff("pkg/calc.py", CALC, CALC_FIXED)))
    assert result.created is False and "was removed" in result.error
    assert list(gh.branches) == ["main"]


def test_a_missing_base_branch_is_reported_and_nothing_is_created() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    result = make_service(gh).create(request_for(make_diff("pkg/calc.py", CALC, CALC_FIXED), base_branch="nope"))
    assert result.created is False and "GitHub API error" in result.error
    assert nothing_was_created(gh)


def test_a_failing_file_read_creates_nothing() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    gh.fail[("GET", "/contents/")] = 500
    result = make_service(gh).create(request_for(make_diff("pkg/calc.py", CALC, CALC_FIXED)))
    assert result.created is False and "GitHub API error" in result.error
    assert nothing_was_created(gh)


def test_the_token_is_sent_as_a_bearer_header_and_never_shown_in_results() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    gh.fail[("PUT", "pkg/calc.py")] = 500
    result = make_service(gh).create(request_for(make_diff("pkg/calc.py", CALC, CALC_FIXED)))
    assert all(r.headers["authorization"] == f"Bearer {TOKEN}" for r in gh.requests)
    assert TOKEN not in result.model_dump_json()


def test_a_failed_validation_makes_no_requests_at_all() -> None:
    gh = FakeGitHub({"pkg/calc.py": CALC})
    result = make_service(gh).create(request_for(make_diff("pkg/calc.py", CALC, CALC_FIXED), validation_passed=False))
    assert result.created is False and "Validation did not pass" in result.skipped_reason
    assert gh.log == []


def test_paths_with_spaces_and_unicode_are_encoded() -> None:
    gh = FakeGitHub({"my dir/café.py": UTIL})
    result = make_service(gh).create(request_for(make_diff("my dir/café.py", UTIL, UTIL + "# x\n")))
    assert result.created is True and gh.text_on(result.branch, "my dir/café.py").endswith("# x\n")


# --- through the /fix/pr route with the reviewed diff -------------------------------------------------------


@pytest.fixture
def route_env(tmp_path: Path) -> Iterator:
    from backend.agent.validation_checks import SyntaxCheck
    from backend.agent.validation_pipeline import ValidationPipeline
    from tests.test_validation_real_repo import FakeProvider

    gh = FakeGitHub({"pkg/calc.py": CALC})
    validator = ValidationPipeline(
        checks=(SyntaxCheck(),), sandbox_parent=tmp_path / "b",
        snapshot_provider=FakeProvider({"pkg/calc.py": CALC}),
    )
    app.dependency_overrides[get_draft_pr_service] = lambda: make_service(gh)
    app.dependency_overrides[get_validation_pipeline] = lambda: validator
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    yield gh, TestClient(app)
    app.dependency_overrides.clear()


def test_the_reviewed_diff_goes_from_the_route_into_the_real_service_unchanged(route_env) -> None:
    gh, client = route_env
    diff = make_diff("pkg/calc.py", CALC, CALC_FIXED)
    proposal = {
        "goal": "fix the add bug", "explanation": "swap - for +", "diff": diff,
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }
    body = client.post("/fix/pr", json={
        "goal": "fix the add bug", "owner": "o", "repo": "r", "proposal": proposal,
    }).json()
    assert body["created"] is True and body["diff_source"] == "reviewed"
    committed = gh.text_on(body["branch"], "pkg/calc.py")
    assert make_diff("pkg/calc.py", CALC, committed) == diff
    assert body["diff_sha256"] == proposal["diff_sha256"]


# --- the patch applier on its own ---------------------------------------------------------------------------------


def _patched(diff: str):
    return list(PatchSet(diff))[0]


def test_applier_matches_difflib_for_ordinary_changes() -> None:
    old = "".join(f"l{i}\n" for i in range(30))
    new = old.replace("l5\n", "five\n").replace("l20\n", "").replace("l29\n", "l29\nlast\n")
    assert apply_patch_to_text(old, _patched(make_diff("f", old, new)), "f") == new


def test_applier_refuses_a_mismatch_and_names_the_line() -> None:
    diff = make_diff("f", "a\nb\nc\n", "a\nB\nc\n")
    with pytest.raises(PatchApplyError, match="does not match the real file"):
        apply_patch_to_text("a\nx\nc\n", _patched(diff), "f")


def test_applier_builds_a_new_file_from_nothing() -> None:
    assert apply_patch_to_text("", _patched(new_file_diff("n.py", "a\nb\n")), "n.py") == "a\nb\n"


def test_applier_deleting_everything_gives_an_empty_text() -> None:
    assert apply_patch_to_text("a\nb\n", _patched(delete_diff("d.py", "a\nb\n")), "d.py") == ""


def test_applier_respects_the_no_newline_marker() -> None:
    diff = "--- a/x\n+++ b/x\n@@ -1,1 +1,2 @@\n a\n+b\n\\ No newline at end of file"
    assert apply_patch_to_text("a\n", _patched(diff), "x") == "a\nb"


@pytest.mark.parametrize("text,eol", [("a\r\nb\r\n", "\r\n"), ("a\nb\n", "\n"), ("a\r\nb\nc\n", "\n"), ("", "\n")])
def test_line_ending_detection(text: str, eol: str) -> None:
    assert dominant_eol(text) == eol
