"""
Concrete ValidationCheck implementations.

Ships:
    - SyntaxCheck: every changed .py file must parse with the ast module
    - ImportCheck: every top-level import in changed files must resolve
      against the sandbox directory (stdlib + files inside the sandbox)
    - NoOpCheck: an intentionally-skipped placeholder used to represent
      "not yet implemented" stages (e.g. real lint / real security).

It stays honest: we run the checks we can trust today. The
pipeline can be extended with more concrete checks later without any
change to the outer contract.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import time
from pathlib import Path
from typing import Protocol

from backend.agent.validation_models import CheckResult


class ValidationCheck(Protocol):
    """Anything that can inspect a sandbox and return a CheckResult."""

    name: str

    def run(self, sandbox_root: Path, changed_files: tuple) -> CheckResult:
        ...


class SyntaxCheck:
    """Every changed .py file must parse cleanly with the ast module."""

    name: str = "syntax"

    def run(self, sandbox_root: Path, changed_files: tuple) -> CheckResult:
        start = time.perf_counter()
        py_files = [f for f in changed_files if str(f).endswith(".py")]
        if not py_files:
            return CheckResult(
                name=self.name, passed=True,
                message="No Python files changed.", duration_ms=0.0,
                skipped=True,
            )

        for rel in py_files:
            full = sandbox_root / rel
            if not full.is_file():
                # Deleted file - nothing to parse.
                continue
            try:
                source = full.read_text(encoding="utf-8")
                ast.parse(source, filename=str(full))
            except SyntaxError as exc:
                return CheckResult(
                    name=self.name, passed=False,
                    message=f"{rel}: SyntaxError line {exc.lineno}: {exc.msg}",
                    duration_ms=(time.perf_counter() - start) * 1000.0,
                )
            except Exception as exc:
                return CheckResult(
                    name=self.name, passed=False,
                    message=f"{rel}: {type(exc).__name__}: {exc}",
                    duration_ms=(time.perf_counter() - start) * 1000.0,
                )

        return CheckResult(
            name=self.name, passed=True,
            message=f"Parsed {len(py_files)} Python file(s) cleanly.",
            duration_ms=(time.perf_counter() - start) * 1000.0,
        )


class ImportCheck:
    """
    Every top-level import in each changed .py file must resolve.

    Resolves imports against:
        - stdlib / installed packages (via importlib.util.find_spec)
        - files that exist inside the sandbox (as sibling modules)

    We deliberately do NOT execute the code - just parse and check
    that names could plausibly be imported. Runtime import errors
    belong to a later stage's actual test execution (future work).
    """

    name: str = "imports"

    def run(self, sandbox_root: Path, changed_files: tuple) -> CheckResult:
        start = time.perf_counter()
        py_files = [f for f in changed_files if str(f).endswith(".py")]
        if not py_files:
            return CheckResult(
                name=self.name, passed=True,
                message="No Python files changed.", duration_ms=0.0,
                skipped=True,
            )

        missing: list = []
        for rel in py_files:
            full = sandbox_root / rel
            if not full.is_file():
                continue
            try:
                tree = ast.parse(full.read_text(encoding="utf-8"))
            except SyntaxError:
                # Syntax will be caught by SyntaxCheck - skip here.
                continue

            for node in tree.body:
                names = self._imported_names(node)
                for name in names:
                    if not self._resolvable(name, sandbox_root):
                        missing.append(f"{rel}: {name}")

        if missing:
            return CheckResult(
                name=self.name, passed=False,
                message=f"Unresolved imports: {', '.join(missing[:5])}"
                + (" ..." if len(missing) > 5 else ""),
                duration_ms=(time.perf_counter() - start) * 1000.0,
            )
        return CheckResult(
            name=self.name, passed=True,
            message=f"All imports resolved across {len(py_files)} file(s).",
            duration_ms=(time.perf_counter() - start) * 1000.0,
        )

    @staticmethod
    def _imported_names(node: ast.AST) -> list:
        if isinstance(node, ast.Import):
            return [alias.name.split(".")[0] for alias in node.names]
        if isinstance(node, ast.ImportFrom) and node.module:
            return [node.module.split(".")[0]]
        return []

    @staticmethod
    def _resolvable(top_level_name: str, sandbox_root: Path) -> bool:
        # 1. Try stdlib / installed packages.
        try:
            if importlib.util.find_spec(top_level_name) is not None:
                return True
        except (ValueError, ModuleNotFoundError, ImportError):
            pass

        # 2. Try sibling module / package inside the sandbox.
        if (sandbox_root / f"{top_level_name}.py").is_file():
            return True
        if (sandbox_root / top_level_name / "__init__.py").is_file():
            return True
        # 3. Also allow top-level dotted package name (e.g. "backend").
        if (sandbox_root / top_level_name).is_dir():
            return True
        return False


class NoOpCheck:
    """An intentionally-skipped check - honest placeholder for future work."""

    def __init__(self, name: str, note: str) -> None:
        self.name = name
        self.note = note

    def run(self, sandbox_root: Path, changed_files: tuple) -> CheckResult:
        return CheckResult(
            name=self.name, passed=True,
            message=f"[skipped] {self.note}", duration_ms=0.0,
            skipped=True,
        )


def default_checks() -> tuple:
    """Return the default suite of checks used by the ValidationPipeline."""
    return (
        SyntaxCheck(),
        ImportCheck(),
        NoOpCheck("lint", "Future: wire ruff / flake8 subprocess."),
        NoOpCheck("tests", "Future: run pytest in a subprocess."),
        NoOpCheck("security", "Future: wire bandit / secret-scan."),
    )