"""
Validation pipeline.

Given a FixProposal, this module:
    1. Materializes a sandbox directory (fresh copy of `base_root`, or
       empty when no base is given).
    2. Applies the proposal's diff to that sandbox using unidiff.
    3. Runs each ValidationCheck against the sandbox.
    4. Returns a structured ValidationResult.

Design notes:
    - We never touch the real project. Everything lands under `tmp_root`.
    - `base_root=None` supports "greenfield" validation for tests where
      the diff creates new files from scratch.
    - Applying the diff is intentionally simple - we handle add / modify
      / delete but skip advanced git-only concepts (renames, binary,
      submodules). Rich diffs failing to apply produce a fatal error
      captured in ValidationResult.error, not a crash.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from backend.agent.fix_models import FixProposal
from backend.agent.validation_checks import ValidationCheck, default_checks
from backend.agent.validation_models import CheckResult, ValidationResult


class ValidationPipeline:
    """Run a sequence of ValidationChecks against a FixProposal."""

    def __init__(
        self,
        checks: tuple | None = None,
        *,
        base_root: Path | None = None,
        sandbox_parent: Path | None = None,
    ) -> None:
        self.checks: tuple = checks if checks is not None else default_checks()
        if not self.checks:
            raise ValueError("At least one ValidationCheck must be provided.")
        self.base_root = base_root
        self.sandbox_parent = sandbox_parent

    def validate(self, proposal: FixProposal) -> ValidationResult:
        """Materialize a sandbox, apply the diff, run checks, aggregate."""
        if not proposal.is_valid:
            return ValidationResult(
                proposal_goal=proposal.goal,
                checks=(),
                passed=False,
                score=0.0,
                error=f"Proposal is not valid: {proposal.validation_error}",
            )

        sandbox = self._make_sandbox()
        try:
            changed = self._apply_diff(sandbox, proposal.diff)
        except Exception as exc:
            self._cleanup(sandbox)
            return ValidationResult(
                proposal_goal=proposal.goal,
                checks=(),
                passed=False,
                score=0.0,
                error=f"Failed to apply diff: {type(exc).__name__}: {exc}",
            )

        try:
            results: list = []
            for check in self.checks:
                results.append(check.run(sandbox, changed))
        finally:
            self._cleanup(sandbox)

        non_skipped = [r for r in results if not r.skipped]
        passed = all(r.passed for r in non_skipped) if non_skipped else True
        score = (
            sum(1 for r in non_skipped if r.passed) / len(non_skipped)
            if non_skipped else 1.0
        )

        return ValidationResult(
            proposal_goal=proposal.goal,
            checks=tuple(results),
            passed=passed,
            score=score,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_sandbox(self) -> Path:
        parent = self.sandbox_parent or Path(tempfile.gettempdir())
        parent.mkdir(parents=True, exist_ok=True)
        sandbox = Path(tempfile.mkdtemp(prefix="agent_sandbox_", dir=str(parent)))
        if self.base_root is not None and self.base_root.is_dir():
            shutil.copytree(self.base_root, sandbox, dirs_exist_ok=True)
        return sandbox

    @staticmethod
    def _cleanup(sandbox: Path) -> None:
        shutil.rmtree(sandbox, ignore_errors=True)

    def _apply_diff(self, sandbox: Path, diff_text: str) -> tuple:
        """Apply a unified diff to the sandbox and return the list of changed rel paths."""
        try:
            from unidiff import PatchSet
        except ImportError as exc:
            raise RuntimeError("unidiff not installed - install requirements.") from exc

        patch = PatchSet(diff_text)
        changed: list = []

        for patched_file in patch:
            rel = self._normalized_path(patched_file)
            target = sandbox / rel

            if patched_file.is_removed_file:
                if target.exists():
                    target.unlink()
                changed.append(rel)
                continue

            # Build the new file line-by-line.
            source_lines: list = []
            if target.exists():
                source_lines = target.read_text(encoding="utf-8").splitlines(keepends=True)

            new_lines = self._reconstruct(source_lines, patched_file)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("".join(new_lines), encoding="utf-8")
            changed.append(rel)

        return tuple(changed)

    @staticmethod
    def _normalized_path(patched_file) -> str:  # type: ignore[no-untyped-def]
        raw = patched_file.path or patched_file.target_file or patched_file.source_file or ""
        raw = str(raw)
        for prefix in ("b/", "a/"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        return raw

    @staticmethod
    def _reconstruct(source_lines: list, patched_file) -> list:  # type: ignore[no-untyped-def]
        """Apply hunks in order to the source, producing new file content."""
        result: list = []
        cursor = 0  # 0-indexed position in source_lines

        for hunk in patched_file:
            source_start_0 = max(hunk.source_start - 1, 0)
            # Copy unchanged lines up to the hunk start.
            while cursor < source_start_0 and cursor < len(source_lines):
                result.append(source_lines[cursor])
                cursor += 1
            # Emit target lines from the hunk.
            for line in hunk:
                # Line categories: source (removed), target (added), context (both).
                if line.is_added or line.is_context:
                    text = line.value
                    if not text.endswith("\n"):
                        text += "\n"
                    result.append(text)
            cursor = source_start_0 + hunk.source_length

        # Copy tail of source that came after the last hunk.
        while cursor < len(source_lines):
            result.append(source_lines[cursor])
            cursor += 1

        return result