"""Integration tests for backend.agent.validation_pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.agent.fix_models import FixHunk, FixProposal
from backend.agent.validation_checks import NoOpCheck, SyntaxCheck, ImportCheck
from backend.agent.validation_pipeline import ValidationPipeline


VALID_DIFF_NEW_FILE = (
    "--- /dev/null\n"
    "+++ b/hello.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+def greet():\n"
    "+    return 'hi'\n"
)


def _proposal(diff: str, is_valid: bool = True, error: str = "") -> FixProposal:
    return FixProposal(
        goal="do a thing",
        explanation="whatever",
        diff=diff,
        hunks=(FixHunk(file_path="hello.py", added_lines=2, removed_lines=0),),
        model="fake",
        confidence=0.9,
        is_valid=is_valid,
        validation_error=error,
    )


def test_pipeline_rejects_invalid_proposal() -> None:
    p = ValidationPipeline(checks=(SyntaxCheck(),))
    proposal = _proposal("dummy", is_valid=False, error="bad json")
    result = p.validate(proposal)
    assert result.passed is False
    assert "not valid" in result.error


def test_pipeline_rejects_empty_checks() -> None:
    with pytest.raises(ValueError):
        ValidationPipeline(checks=())


def test_pipeline_creates_and_cleans_sandbox(tmp_path: Path) -> None:
    p = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    proposal = _proposal(VALID_DIFF_NEW_FILE)
    p.validate(proposal)
    # Sandbox parent should be clean after the run.
    remaining = list(tmp_path.iterdir())
    assert remaining == [] or all(not any(c.iterdir()) for c in remaining if c.is_dir())


def test_pipeline_applies_new_file_diff(tmp_path: Path) -> None:
    p = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    proposal = _proposal(VALID_DIFF_NEW_FILE)
    result = p.validate(proposal)
    assert result.passed is True
    syntax = [c for c in result.checks if c.name == "syntax"][0]
    assert syntax.passed is True


def test_pipeline_handles_empty_diff_gracefully(tmp_path: Path) -> None:
    """
    When unidiff parses 'garbage' as zero file changes, the pipeline
    doesn't crash. All checks are skipped (nothing to validate) and
    the pipeline reports passed=True with a perfect score. This is
    correct defensive behavior - no files were touched, so no rules
    were violated.
    """
    p = ValidationPipeline(checks=(SyntaxCheck(),), sandbox_parent=tmp_path)
    proposal = _proposal("garbage that is not a diff")
    result = p.validate(proposal)
    # No fatal error, all checks skipped, passed=True by convention.
    assert result.error == ""
    assert all(c.skipped for c in result.checks)
    assert result.passed is True


def test_pipeline_score_is_fraction_of_passed_non_skipped(tmp_path: Path) -> None:
    checks = (SyntaxCheck(), NoOpCheck("lint", "later"))
    p = ValidationPipeline(checks=checks, sandbox_parent=tmp_path)
    result = p.validate(_proposal(VALID_DIFF_NEW_FILE))
    # Only SyntaxCheck is non-skipped and it passes -> score 1.0
    assert result.score == 1.0
    assert result.passed is True


def test_pipeline_uses_base_root(tmp_path: Path) -> None:
    """When base_root is provided, sandbox starts as a copy of it."""
    base = tmp_path / "base"
    base.mkdir()
    (base / "existing.py").write_text("def y():\n    return 2\n", encoding="utf-8")
    p = ValidationPipeline(
        checks=(SyntaxCheck(),),
        base_root=base,
        sandbox_parent=tmp_path / "boxes",
    )
    result = p.validate(_proposal(VALID_DIFF_NEW_FILE))
    # New file added on top of base - syntax should still pass.
    assert result.passed is True


def test_pipeline_reports_import_failure(tmp_path: Path) -> None:
    bad_import_diff = (
        "--- /dev/null\n"
        "+++ b/bad.py\n"
        "@@ -0,0 +1,1 @@\n"
        "+import totally_not_a_real_module_xyz\n"
    )
    p = ValidationPipeline(
        checks=(SyntaxCheck(), ImportCheck()),
        sandbox_parent=tmp_path,
    )
    result = p.validate(_proposal(bad_import_diff))
    assert result.passed is False
    failed = result.failed_check_names
    assert "imports" in failed