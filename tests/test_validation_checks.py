"""Unit tests for backend.agent.validation_checks."""

from __future__ import annotations

from pathlib import Path

from backend.agent.validation_checks import (
    ImportCheck, NoOpCheck, SyntaxCheck, default_checks,
)


def _make_file(root: Path, rel: str, source: str) -> None:
    full = root / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(source, encoding="utf-8")


def test_syntax_check_passes_clean_file(tmp_path: Path) -> None:
    _make_file(tmp_path, "app.py", "def x():\n    return 1\n")
    result = SyntaxCheck().run(tmp_path, ("app.py",))
    assert result.passed is True
    assert result.skipped is False


def test_syntax_check_fails_broken_file(tmp_path: Path) -> None:
    _make_file(tmp_path, "bad.py", "def x(:\n    return 1\n")
    result = SyntaxCheck().run(tmp_path, ("bad.py",))
    assert result.passed is False
    assert "SyntaxError" in result.message


def test_syntax_check_skips_when_no_py(tmp_path: Path) -> None:
    _make_file(tmp_path, "README.md", "hi")
    result = SyntaxCheck().run(tmp_path, ("README.md",))
    assert result.skipped is True
    assert result.passed is True


def test_import_check_passes_stdlib(tmp_path: Path) -> None:
    _make_file(tmp_path, "app.py", "import os\nimport json\n")
    result = ImportCheck().run(tmp_path, ("app.py",))
    assert result.passed is True


def test_import_check_passes_sibling_module(tmp_path: Path) -> None:
    _make_file(tmp_path, "helper.py", "def x(): return 1\n")
    _make_file(tmp_path, "app.py", "import helper\nprint(helper.x())\n")
    result = ImportCheck().run(tmp_path, ("app.py",))
    assert result.passed is True


def test_import_check_fails_missing_top_level(tmp_path: Path) -> None:
    _make_file(tmp_path, "app.py", "import totally_not_a_real_module_xyz\n")
    result = ImportCheck().run(tmp_path, ("app.py",))
    assert result.passed is False
    assert "totally_not_a_real_module_xyz" in result.message


def test_import_check_skips_when_no_py(tmp_path: Path) -> None:
    result = ImportCheck().run(tmp_path, ("README.md",))
    assert result.skipped is True


def test_import_check_ignores_files_with_syntax_errors(tmp_path: Path) -> None:
    _make_file(tmp_path, "bad.py", "def x(:\n    pass\n")
    result = ImportCheck().run(tmp_path, ("bad.py",))
    # ImportCheck doesn't crash; it skips broken files silently.
    assert result.passed is True


def test_no_op_check_reports_skipped(tmp_path: Path) -> None:
    check = NoOpCheck("lint", "not yet")
    result = check.run(tmp_path, ("app.py",))
    assert result.skipped is True
    assert result.passed is True
    assert "not yet" in result.message


def test_default_checks_returns_five() -> None:
    checks = default_checks()
    assert len(checks) == 5
    names = tuple(c.name for c in checks)
    assert names == ("syntax", "imports", "lint", "tests", "security")