"""Loop 3 tests: validating a patch against the real repository files (OFFLINE).

A fake snapshot provider stands in for GitHub. These tests prove the patch
is checked against the real file text and that bad patches fail with the file
and the reason. They do not call GitHub.
"""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from backend.agent.fix_models import FixProposal
from backend.agent.validation_checks import ImportCheck, SyntaxCheck
from backend.agent.validation_models import RepoTarget
from backend.agent.validation_pipeline import ValidationPipeline
from backend.github.repo_snapshot import RepoSnapshot, SnapshotError


COMMIT = "abc1234" + "0" * 33

CALC = (
    "import os\n"
    "\n"
    "def add(a, b):\n"
    "    return a - b\n"
    "\n"
    "def sub(a, b):\n"
    "    return a - b\n"
)
CALC_FIXED = CALC.replace("def add(a, b):\n    return a - b", "def add(a, b):\n    return a + b")


def _diff(path: str, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    ))


def _proposal(diff: str) -> FixProposal:
    return FixProposal(
        goal="fix add", explanation="e", diff=diff, model="m", confidence=0.9, is_valid=True,
    )


@dataclass
class FakeProvider:
    """Serves files from a dict, like GitHub would, and records every call."""

    repo_files: dict
    commit: str = COMMIT
    fail: Exception | None = None
    calls: list = field(default_factory=list)

    def fetch(self, owner, repo, ref, paths):
        self.calls.append({"owner": owner, "repo": repo, "ref": ref, "paths": list(paths)})
        if self.fail:
            raise self.fail
        files = {p: self.repo_files[p] for p in paths if p in self.repo_files}
        missing = tuple(p for p in paths if p not in self.repo_files)
        tree = tuple(sorted(p for p in self.repo_files if p.endswith(".py")))
        return RepoSnapshot(commit=self.commit, files=files, missing=missing, tree_paths=tree)


REPO = RepoTarget(owner="o", repo="r", ref="main")


def _pipeline(tmp_path: Path, provider, checks=None) -> ValidationPipeline:
    return ValidationPipeline(
        checks=checks or (SyntaxCheck(), ImportCheck()),
        sandbox_parent=tmp_path / "boxes",
        snapshot_provider=provider,
    )


def _leftovers(tmp_path: Path) -> list:
    boxes = tmp_path / "boxes"
    return list(boxes.iterdir()) if boxes.exists() else []


# --- known good / known bad ---------------------------------------------


def test_known_good_patch_passes_against_the_real_file(tmp_path: Path) -> None:
    provider = FakeProvider({"pkg/calc.py": CALC})
    result = _pipeline(tmp_path, provider).validate(
        _proposal(_diff("pkg/calc.py", CALC, CALC_FIXED)), repo=REPO,
    )
    assert result.error == ""
    assert result.passed is True
    assert result.base_commit == COMMIT
    assert "o/r@abc1234" in result.base_description
    assert "1 file(s) fetched" in result.base_description


def test_known_bad_patch_fails_and_names_the_file_and_line(tmp_path: Path) -> None:
    # The patch was written for different code than the real file holds.
    other = CALC.replace("a - b", "a * b")
    provider = FakeProvider({"pkg/calc.py": CALC})
    result = _pipeline(tmp_path, provider).validate(
        _proposal(_diff("pkg/calc.py", other, other.replace("a * b", "a + b", 1))), repo=REPO,
    )
    assert result.passed is False
    assert "pkg/calc.py" in result.error
    assert "does not match the real file" in result.error
    assert "Line 4" in result.error
    assert result.base_commit == COMMIT


def test_syntax_error_in_a_good_fit_patch_still_fails(tmp_path: Path) -> None:
    broken = CALC.replace("return a - b\n\ndef sub", "return a +\n\ndef sub")
    provider = FakeProvider({"pkg/calc.py": CALC})
    result = _pipeline(tmp_path, provider).validate(
        _proposal(_diff("pkg/calc.py", CALC, broken)), repo=REPO,
    )
    assert result.error == ""
    assert result.passed is False
    assert "syntax" in result.failed_check_names


def test_patch_with_wrong_line_numbers_still_applies_when_the_lines_match(tmp_path: Path) -> None:
    diff = _diff("pkg/calc.py", CALC, CALC_FIXED).replace("@@ -1,", "@@ -40,", 1)
    assert "@@ -40," in diff
    provider = FakeProvider({"pkg/calc.py": CALC})
    result = _pipeline(tmp_path, provider).validate(_proposal(diff), repo=REPO)
    assert result.error == ""
    assert result.passed is True


def test_fix_for_the_second_function_is_applied_to_the_right_place(tmp_path: Path) -> None:
    fixed = CALC.replace("def sub(a, b):\n    return a - b", "def sub(a, b):\n    return a - b - 0")
    provider = FakeProvider({"pkg/calc.py": CALC})
    result = _pipeline(tmp_path, provider).validate(
        _proposal(_diff("pkg/calc.py", CALC, fixed)), repo=REPO,
    )
    assert result.passed is True


# --- the files must really exist ----------------------------------------


def test_changing_a_file_that_does_not_exist_at_that_commit_fails(tmp_path: Path) -> None:
    provider = FakeProvider({"pkg/other.py": "x = 1\n"})
    result = _pipeline(tmp_path, provider).validate(
        _proposal(_diff("pkg/calc.py", CALC, CALC_FIXED)), repo=REPO,
    )
    assert result.passed is False
    assert "pkg/calc.py" in result.error
    assert "does not exist at commit abc1234" in result.error


def test_creating_a_file_that_already_exists_fails(tmp_path: Path) -> None:
    provider = FakeProvider({"pkg/calc.py": CALC})
    new_file = "--- /dev/null\n+++ b/pkg/calc.py\n@@ -0,0 +1,1 @@\n+x = 1\n"
    result = _pipeline(tmp_path, provider).validate(_proposal(new_file), repo=REPO)
    assert result.passed is False
    assert "already exists" in result.error


def test_creating_a_new_file_passes(tmp_path: Path) -> None:
    provider = FakeProvider({"pkg/calc.py": CALC})
    new_file = "--- /dev/null\n+++ b/pkg/new.py\n@@ -0,0 +1,2 @@\n+def f():\n+    return 1\n"
    result = _pipeline(tmp_path, provider).validate(_proposal(new_file), repo=REPO)
    assert result.error == ""
    assert result.passed is True
    assert provider.calls[0]["paths"] == []


def test_deleting_an_existing_file_passes_and_deleting_a_missing_one_fails(tmp_path: Path) -> None:
    provider = FakeProvider({"pkg/old.py": "x = 1\n"})
    delete = "--- a/pkg/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x = 1\n"
    assert _pipeline(tmp_path, provider).validate(_proposal(delete), repo=REPO).passed is True
    missing = "--- a/pkg/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x = 1\n"
    result = _pipeline(tmp_path, provider).validate(_proposal(missing), repo=REPO)
    assert result.passed is False
    assert "pkg/gone.py" in result.error


def test_only_the_touched_files_are_requested_from_the_provider(tmp_path: Path) -> None:
    provider = FakeProvider({"pkg/calc.py": CALC, "pkg/other.py": "x = 1\n"})
    _pipeline(tmp_path, provider).validate(
        _proposal(_diff("pkg/calc.py", CALC, CALC_FIXED)), repo=REPO,
    )
    assert provider.calls == [
        {"owner": "o", "repo": "r", "ref": "main", "paths": ["pkg/calc.py"]},
    ]


# --- failures are loud, never silent -------------------------------------


@pytest.mark.parametrize("error", [
    SnapshotError("Not found on GitHub: /repos/o/r"),
    RuntimeError("socket closed"),
])
def test_provider_failure_fails_validation_and_does_not_fall_back_to_empty_folder(
    tmp_path: Path, error: Exception,
) -> None:
    provider = FakeProvider({}, fail=error)
    result = _pipeline(tmp_path, provider).validate(
        _proposal(_diff("pkg/calc.py", CALC, CALC_FIXED)), repo=REPO,
    )
    assert result.passed is False
    assert result.checks == ()
    assert "Could not get the repository files" in result.error
    assert _leftovers(tmp_path) == []


def test_no_repo_given_says_so(tmp_path: Path) -> None:
    new_file = "--- /dev/null\n+++ b/hello.py\n@@ -0,0 +1 @@\n+x = 1\n"
    result = _pipeline(tmp_path, FakeProvider({})).validate(_proposal(new_file))
    assert result.passed is True
    assert result.base_description == "empty folder (no repository was given)"
    assert result.base_commit == ""


def test_repo_given_but_no_provider_says_so(tmp_path: Path) -> None:
    new_file = "--- /dev/null\n+++ b/hello.py\n@@ -0,0 +1 @@\n+x = 1\n"
    result = _pipeline(tmp_path, None).validate(_proposal(new_file), repo=REPO)
    assert "no repository access is configured" in result.base_description


# --- safety: nothing outside the sandbox, nothing left behind -------------


@pytest.mark.parametrize("path", ["../evil.py", "a/../../evil.py", "/tmp/evil.py"])
def test_paths_that_leave_the_repo_are_refused(tmp_path: Path, path: str) -> None:
    diff = f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+x = 1\n"
    for provider in (FakeProvider({}), None):
        result = _pipeline(tmp_path, provider).validate(_proposal(diff), repo=REPO)
        assert result.passed is False
        assert "Unsafe file path" in result.error
    assert not (tmp_path / "evil.py").exists()
    assert not Path("/tmp/evil.py").exists()


def test_sandbox_is_deleted_after_pass_fail_and_error(tmp_path: Path) -> None:
    provider = FakeProvider({"pkg/calc.py": CALC})
    pipe = _pipeline(tmp_path, provider)
    pipe.validate(_proposal(_diff("pkg/calc.py", CALC, CALC_FIXED)), repo=REPO)
    pipe.validate(_proposal(_diff("pkg/calc.py", CALC.replace("a - b", "q"), CALC)), repo=REPO)
    pipe.validate(_proposal("--- a/pkg/zzz.py\n+++ b/pkg/zzz.py\n@@ -1 +1 @@\n-a\n+b\n"), repo=REPO)
    assert _leftovers(tmp_path) == []


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        h.update(str(p.relative_to(root)).encode())
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def test_original_folder_is_unchanged_by_validation(tmp_path: Path) -> None:
    base = tmp_path / "base"
    (base / "pkg").mkdir(parents=True)
    (base / "pkg" / "calc.py").write_text(CALC, encoding="utf-8")
    before = _tree_hash(base)
    pipe = ValidationPipeline(
        checks=(SyntaxCheck(),), base_root=base, sandbox_parent=tmp_path / "boxes",
    )
    good = pipe.validate(_proposal(_diff("pkg/calc.py", CALC, CALC_FIXED)))
    bad = pipe.validate(_proposal(_diff("pkg/calc.py", CALC.replace("a - b", "q"), CALC)))
    assert good.passed is True
    assert bad.passed is False and "pkg/calc.py" in bad.error
    assert _tree_hash(base) == before
    assert _leftovers(tmp_path) == []


def test_local_base_root_also_rejects_a_patch_for_a_missing_file(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    pipe = ValidationPipeline(checks=(SyntaxCheck(),), base_root=base, sandbox_parent=tmp_path / "boxes")
    result = pipe.validate(_proposal(_diff("pkg/calc.py", CALC, CALC_FIXED)))
    assert result.passed is False
    assert "does not exist" in result.error


# --- import check on a real repo layout ----------------------------------


def test_relative_imports_resolve_against_the_repo_tree(tmp_path: Path) -> None:
    repo_files = {"pkg/calc.py": CALC, "pkg/helpers.py": "def h():\n    return 1\n"}
    provider = FakeProvider(repo_files)
    fixed = CALC_FIXED.replace("import os\n", "import os\nfrom .helpers import h\n")
    ok = _pipeline(tmp_path, provider).validate(
        _proposal(_diff("pkg/calc.py", CALC, fixed)), repo=REPO,
    )
    assert ok.passed is True

    broken = CALC_FIXED.replace("import os\n", "import os\nfrom .nothing_here import h\n")
    bad = _pipeline(tmp_path, provider).validate(
        _proposal(_diff("pkg/calc.py", CALC, broken)), repo=REPO,
    )
    assert bad.passed is False
    assert "imports" in bad.failed_check_names


def test_src_layout_packages_resolve(tmp_path: Path) -> None:
    repo_files = {
        "src/zzdemo_pkg/__init__.py": "",
        "tools/run.py": "x = 1\n",
    }
    provider = FakeProvider(repo_files)
    new_text = "import zzdemo_pkg\nx = 1\n"
    ok = _pipeline(tmp_path, provider).validate(
        _proposal(_diff("tools/run.py", "x = 1\n", new_text)), repo=REPO,
    )
    assert ok.passed is True

    provider_without = FakeProvider({"tools/run.py": "x = 1\n"})
    bad = _pipeline(tmp_path, provider_without).validate(
        _proposal(_diff("tools/run.py", "x = 1\n", new_text)), repo=REPO,
    )
    assert bad.passed is False


def test_second_line_of_defence_refuses_paths_outside_the_sandbox(tmp_path: Path) -> None:
    from backend.agent.validation_pipeline import PatchApplyError

    box = tmp_path / "box"
    box.mkdir()
    with pytest.raises(PatchApplyError, match="outside the repository"):
        ValidationPipeline._inside(box, "../escape.py")
    assert ValidationPipeline._inside(box, "pkg/ok.py") == (box / "pkg" / "ok.py").resolve()
