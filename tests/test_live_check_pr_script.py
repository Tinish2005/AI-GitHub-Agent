"""Offline tests of scripts/live_check_pr.py (a fake GitHub, no network)."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.test_pr_modify import BASE_SHA, FakeGitHub, make_service

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "live_check_pr.py"
NOW = datetime(2026, 10, 21, 12, 0, 0, tzinfo=timezone.utc)
README = "# Demo\n\nHello.\n"


def _load():
    spec = importlib.util.spec_from_file_location("live_check_pr", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _levels(results: list) -> list:
    return [level for level, _ in results]


def test_the_check_passes_against_a_well_behaved_repo_and_leaves_the_pr_open() -> None:
    gh = FakeGitHub({"README.md": README})
    results = _load().run_check(make_service(gh), "o", "r", "main", "README.md", cleanup=False, now=NOW)
    assert "FAIL" not in _levels(results) and _levels(results).count("PASS") == 5
    assert results[-1][0] == "INFO" and "Not cleaned up" in results[-1][1]
    branch = next(b for b in gh.branches if b.startswith("agent/"))
    assert gh.text_on(branch, "README.md") == README + "<!-- agent live check 2026-10-21 12:00:00Z -->\n"
    assert gh.pr_state == "open" and gh.branches["main"] == BASE_SHA


def test_cleanup_closes_the_pr_and_deletes_the_branch() -> None:
    gh = FakeGitHub({"README.md": README})
    results = _load().run_check(make_service(gh), "o", "r", "main", "README.md", cleanup=True, now=NOW)
    assert "FAIL" not in _levels(results)
    assert gh.pr_state == "closed" and list(gh.branches) == ["main"]


def test_a_python_file_gets_a_hash_comment_and_crlf_is_kept() -> None:
    gh = FakeGitHub({"app.py": "x = 1\r\ny = 2\r\n"})
    results = _load().run_check(make_service(gh), "o", "r", "main", "app.py", cleanup=False, now=NOW)
    assert "FAIL" not in _levels(results)
    branch = next(b for b in gh.branches if b.startswith("agent/"))
    assert gh.text_on(branch, "app.py") == "x = 1\r\ny = 2\r\n# agent live check 2026-10-21 12:00:00Z\r\n"


def test_a_file_without_a_final_newline_is_handled() -> None:
    gh = FakeGitHub({"README.md": "no newline"})
    results = _load().run_check(make_service(gh), "o", "r", "main", "README.md", cleanup=False, now=NOW)
    assert "FAIL" not in _levels(results)


def test_a_missing_file_is_reported_without_creating_anything() -> None:
    gh = FakeGitHub({"other.md": README})
    results = _load().run_check(make_service(gh), "o", "r", "main", "README.md", cleanup=False, now=NOW)
    assert _levels(results) == ["FAIL"] and "--file" in results[0][1]
    assert gh.prs == [] and list(gh.branches) == ["main"]


def test_a_refused_write_is_reported_as_a_failure() -> None:
    gh = FakeGitHub({"README.md": README})
    gh.fail[("POST", "/git/refs")] = 403
    results = _load().run_check(make_service(gh), "o", "r", "main", "README.md", cleanup=False, now=NOW)
    assert _levels(results) == ["FAIL"] and "not created" in results[0][1]


def test_the_script_refuses_to_run_without_yes(capsys: pytest.CaptureFixture) -> None:
    assert _load().main(["live_check_pr", "--owner", "o", "--repo", "r"]) == 2
    assert "writes to GitHub" in capsys.readouterr().out
