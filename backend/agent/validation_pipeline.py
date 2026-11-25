"""
Validation pipeline.

Given a FixProposal, this module:
    1. Materializes a sandbox directory. When a repository is given, the
       real files the patch touches are fetched from GitHub at one exact
       commit and placed in the sandbox. With only `base_root`, the sandbox
       is a copy of that folder. With neither, it starts empty.
    2. Applies the proposal's diff to that sandbox. When the sandbox holds
       the real file, every hunk must match the real lines, otherwise the
       patch fails and the message names the file and the line.
    3. Runs each ValidationCheck against the sandbox.
    4. Returns a structured ValidationResult.

Design notes:
    - We never touch the real project. Everything lands under `tmp_root`
      and is deleted afterwards.
    - With no repository and no `base_root` the sandbox is empty ("greenfield"
      validation). The result says so in `base_description`. A patch that
      modifies a file cannot be properly checked in that mode.
    - Applying the diff handles add / modify / delete but skips advanced
      git-only concepts (renames, binary, submodules). Diffs that fail to
      apply produce a fatal error in ValidationResult.error, not a crash.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from backend.agent.fix_models import FixProposal
from backend.agent.validation_checks import ValidationCheck, default_checks
from backend.agent.validation_models import CheckResult, RepoTarget, ValidationResult
from backend.github.repo_snapshot import SnapshotError, safe_repo_path


class PatchApplyError(Exception):
    """The patch does not fit the files it is applied to. The message is safe to show."""


class ValidationPipeline:
    """Run a sequence of ValidationChecks against a FixProposal."""

    def __init__(
        self,
        checks: tuple | None = None,
        *,
        base_root: Path | None = None,
        sandbox_parent: Path | None = None,
        snapshot_provider=None,
    ) -> None:
        self.checks: tuple = checks if checks is not None else default_checks()
        if not self.checks:
            raise ValueError("At least one ValidationCheck must be provided.")
        self.base_root = base_root
        self.sandbox_parent = sandbox_parent
        self.snapshot_provider = snapshot_provider

    def validate(self, proposal: FixProposal, *, repo: RepoTarget | None = None) -> ValidationResult:
        """Materialize a sandbox, apply the diff, run checks, aggregate."""
        if not proposal.is_valid:
            return ValidationResult(
                proposal_goal=proposal.goal,
                checks=(),
                passed=False,
                score=0.0,
                error=f"Proposal is not valid: {proposal.validation_error}",
                diff_sha256=proposal.diff_sha256,
            )

        try:
            targets = self._parse_targets(proposal.diff)
        except Exception as exc:
            return self._fatal(proposal, f"Failed to apply diff: {type(exc).__name__}: {exc}")

        snapshot = None
        base_commit = ""
        if repo is not None and self.snapshot_provider is not None:
            to_fetch = [rel for patched_file, rel in targets if not patched_file.is_added_file]
            try:
                snapshot = self.snapshot_provider.fetch(repo.owner, repo.repo, repo.ref, to_fetch)
            except SnapshotError as exc:
                return self._fatal(proposal, f"Could not get the repository files: {exc}")
            except Exception as exc:
                return self._fatal(
                    proposal, f"Could not get the repository files: {type(exc).__name__}: {exc}"
                )
            base_commit = snapshot.commit
            problem = self._check_against_tree(targets, snapshot)
            if problem:
                return self._fatal(proposal, problem, base_commit=base_commit,
                                   base_description=self._describe(repo, snapshot))
            base_description = self._describe(repo, snapshot)
        elif repo is not None:
            base_description = "empty folder (no repository access is configured on this server)"
        elif self.base_root is not None and self.base_root.is_dir():
            base_description = "a copy of a local folder"
        else:
            base_description = "empty folder (no repository was given)"

        strict_missing = snapshot is not None or (
            self.base_root is not None and self.base_root.is_dir()
        )

        sandbox = self._make_sandbox()
        try:
            if snapshot is not None:
                self._fill_from_snapshot(sandbox, snapshot)
            changed = self._apply_patch(sandbox, targets, strict_missing=strict_missing)
        except PatchApplyError as exc:
            self._cleanup(sandbox)
            return self._fatal(
                proposal, f"The patch does not fit the repository: {exc}",
                base_commit=base_commit, base_description=base_description,
            )
        except Exception as exc:
            self._cleanup(sandbox)
            return self._fatal(
                proposal, f"Failed to apply diff: {type(exc).__name__}: {exc}",
                base_commit=base_commit, base_description=base_description,
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
            base_commit=base_commit,
            base_description=base_description,
            diff_sha256=proposal.diff_sha256,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fatal(proposal: FixProposal, error: str, *, base_commit: str = "",
               base_description: str = "") -> ValidationResult:
        return ValidationResult(
            proposal_goal=proposal.goal,
            checks=(),
            passed=False,
            score=0.0,
            error=error,
            base_commit=base_commit,
            base_description=base_description,
            diff_sha256=proposal.diff_sha256,
        )

    @staticmethod
    def _describe(repo: RepoTarget, snapshot) -> str:
        return (
            f"{repo.owner}/{repo.repo}@{snapshot.commit[:7]} "
            f"({len(snapshot.files)} file(s) fetched from GitHub)"
        )

    def _parse_targets(self, diff_text: str) -> list:
        """Parse the diff into (patched_file, safe relative path) pairs."""
        try:
            from unidiff import PatchSet
        except ImportError as exc:
            raise RuntimeError("unidiff not installed - install requirements.") from exc
        patch = PatchSet(diff_text)
        targets: list = []
        for patched_file in patch:
            targets.append((patched_file, safe_repo_path(self._normalized_path(patched_file))))
        return targets

    @staticmethod
    def _check_against_tree(targets: list, snapshot) -> str:
        """Return a problem description when the patch cannot match the real repo."""
        missing = set(snapshot.missing)
        existing = set(snapshot.tree_paths)
        for patched_file, rel in targets:
            if patched_file.is_added_file:
                if rel in existing:
                    return (
                        f"The patch creates {rel}, but that file already exists "
                        f"at commit {snapshot.commit[:7]}."
                    )
            elif rel in missing:
                return (
                    f"The patch changes {rel}, but that file does not exist "
                    f"at commit {snapshot.commit[:7]}."
                )
        return ""

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

    @staticmethod
    def _inside(sandbox: Path, rel: str) -> Path:
        """Return sandbox/rel, refusing any path that would leave the sandbox."""
        root = sandbox.resolve()
        target = (root / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise PatchApplyError(f"{rel}: the path is outside the repository.") from exc
        return target

    def _fill_from_snapshot(self, sandbox: Path, snapshot) -> None:
        """Write the fetched files, plus empty placeholders for the other .py files.

        The placeholders only let the import check see which modules exist.
        """
        for rel, text in snapshot.files.items():
            target = self._inside(sandbox, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        for rel in snapshot.tree_paths:
            if rel in snapshot.files:
                continue
            target = self._inside(sandbox, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()

    def _apply_patch(self, sandbox: Path, targets: list, *, strict_missing: bool) -> tuple:
        """Apply the parsed diff to the sandbox; return the changed relative paths."""
        changed: list = []

        for patched_file, rel in targets:
            target = self._inside(sandbox, rel)

            if patched_file.is_removed_file:
                if target.exists():
                    target.unlink()
                elif strict_missing:
                    raise PatchApplyError(f"{rel}: the patch deletes this file, but it does not exist.")
                changed.append(rel)
                continue

            # Build the new file line-by-line.
            source_lines: list = []
            exists = target.exists()
            if exists and not patched_file.is_added_file:
                source_lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
            elif not patched_file.is_added_file and strict_missing:
                raise PatchApplyError(f"{rel}: the patch changes this file, but it does not exist.")

            strict = exists and not patched_file.is_added_file
            if strict:
                new_lines = self._reconstruct_strict(source_lines, patched_file, rel)
            else:
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
        """Apply hunks by line number only (used when there is no real file to check)."""
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

    @classmethod
    def _reconstruct_strict(cls, source_lines: list, patched_file, rel: str) -> list:  # type: ignore[no-untyped-def]
        """Apply hunks to a real file. Every context and removed line must match.

        Like the `patch` tool, a hunk may sit a few lines away from the line
        number in its header, as long as its lines match exactly.
        """
        result: list = []
        cursor = 0

        for hunk in patched_file:
            wanted = [ln.value for ln in hunk if ln.is_context or ln.is_removed]
            if hunk.source_length == 0:
                expected = hunk.source_start  # pure insertion: after this line
            else:
                expected = max(hunk.source_start - 1, 0)
            pos = cls._locate(source_lines, wanted, expected, cursor)
            if pos is None:
                raise PatchApplyError(cls._mismatch(rel, hunk, source_lines, wanted, expected))

            result.extend(source_lines[cursor:pos])
            for line in hunk:
                if line.is_added or line.is_context:
                    text = line.value
                    if not text.endswith("\n"):
                        text += "\n"
                    result.append(text)
            cursor = pos + len(wanted)

        result.extend(source_lines[cursor:])
        return result

    @staticmethod
    def _locate(lines: list, wanted: list, expected: int, cursor: int):
        n = len(wanted)
        if n == 0:
            return min(max(expected, cursor), len(lines))

        def matches(i: int) -> bool:
            if i < cursor or i + n > len(lines):
                return False
            return all(
                lines[i + k].rstrip("\r\n") == wanted[k].rstrip("\r\n") for k in range(n)
            )

        if matches(expected):
            return expected
        best = None
        for i in range(cursor, len(lines) - n + 1):
            if matches(i):
                distance = abs(i - expected)
                if best is None or distance < best[0]:
                    best = (distance, i)
        return best[1] if best else None

    @staticmethod
    def _mismatch(rel: str, hunk, lines: list, wanted: list, expected: int) -> str:
        def short(text: str) -> str:
            text = text.rstrip("\r\n")
            return repr(text if len(text) <= 80 else text[:77] + "...")

        for k, want in enumerate(wanted):
            idx = expected + k
            have = lines[idx] if idx < len(lines) else None
            if have is None or have.rstrip("\r\n") != want.rstrip("\r\n"):
                found = short(have) if have is not None else "the end of the file"
                return (
                    f"{rel}: the change at line {hunk.source_start} does not match the real file. "
                    f"Line {idx + 1} should be {short(want)} but the file has {found}."
                )
        return f"{rel}: the change at line {hunk.source_start} does not match the real file."
