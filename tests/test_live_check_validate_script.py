"""Offline test of scripts/live_check_validate.py (a fake GitHub, no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from backend.github.repo_snapshot import RepoSnapshot, SnapshotError

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "live_check_validate.py"

SOURCE = "import os\n\n\nclass Thing:\n    pass\n\n\ndef run():\n    return os.sep\n"


def _load():
    spec = importlib.util.spec_from_file_location("live_check_validate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Provider:
    def fetch(self, owner, repo, ref, paths):
        if repo != "r":
            raise SnapshotError("Not found on GitHub: /repos/o/x")
        files = {p: SOURCE for p in paths if p == "pkg/mod.py"}
        missing = tuple(p for p in paths if p not in files)
        return RepoSnapshot(commit="c" * 40, files=files, missing=missing, tree_paths=("pkg/mod.py",))


def test_all_four_cases_behave_correctly_with_a_well_behaved_repo() -> None:
    cases = _load().run_cases(_Provider(), "o", "r", "pkg/mod.py")
    assert [c["ok"] for c in cases] == [True, True, True, True], cases


def test_script_reports_a_problem_when_the_file_is_missing() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="was not found"):
        _load().run_cases(_Provider(), "o", "r", "pkg/other.py")
