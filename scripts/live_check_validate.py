"""LIVE check for Loop 3: validate patches against a real GitHub repo.

Run from the project root, in PowerShell:

    python scripts\\live_check_validate.py

Optional: python scripts\\live_check_validate.py psf requests src/requests/sessions.py

It uses the real GitHub API. A GitHub token is optional (it raises the rate
limit from 60 to 5000 requests an hour). It is read from the GITHUB_TOKEN
environment variable, or from your .env file. The token is never printed.
This check makes about 9 GitHub requests and never changes anything.

It builds its own patches from the live file, so it keeps working when the
repo changes:
    1. known-good patch (adds a comment)      -> must PASS
    2. known-bad patch (wrong context lines)  -> must FAIL and name the file
    3. patch for a file that does not exist   -> must FAIL and say so
    4. a repo that does not exist             -> must FAIL with a clear error

Exit code: 0 = all four behaved correctly, 1 = at least one did not, 2 = setup problem.
"""

from __future__ import annotations

import difflib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NO_SUCH_REPO = "__no_such_repo_for_validation_check__"


def _diff(path: str, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    ))


def _proposal(diff: str):
    from backend.agent.fix_models import FixProposal

    return FixProposal(
        goal="live check", explanation="live check", diff=diff,
        model="live-check", confidence=1.0, is_valid=True,
    )


def _pick_line(text: str) -> int:
    """Index of a line that starts a class or function (safe to add a comment to)."""
    for i, line in enumerate(text.splitlines()):
        if line.startswith(("class ", "def ")) and line.rstrip().endswith(":"):
            return i
    raise RuntimeError("No class or def line found in the file; pick another file.")


def run_cases(provider, owner: str, repo: str, path: str) -> list:
    """Run the four cases. Returns a list of dicts: name, expected, ok, detail."""
    from backend.agent.validation_models import RepoTarget
    from backend.agent.validation_pipeline import ValidationPipeline

    snap = provider.fetch(owner, repo, None, [path])
    if path not in snap.files:
        raise RuntimeError(f"{path} was not found in {owner}/{repo}; pass another file path.")
    text = snap.files[path]
    lines = text.splitlines(keepends=True)
    i = _pick_line(text)

    good_lines = list(lines)
    good_lines[i] = lines[i].rstrip("\r\n") + "  # live check\n"
    bad_old = list(lines)
    bad_old[i] = "THIS LINE IS NOT IN THE REAL FILE:\n"

    pipeline = ValidationPipeline(snapshot_provider=provider)
    target = RepoTarget(owner=owner, repo=repo)
    cases = []

    r = pipeline.validate(_proposal(_diff(path, text, "".join(good_lines))), repo=target)
    cases.append({
        "name": "known-good patch", "expected": "passed=True",
        "ok": r.passed and not r.error,
        "detail": f"passed={r.passed} commit={r.base_commit[:7]} {r.error}".strip(),
    })

    r = pipeline.validate(_proposal(_diff(path, "".join(bad_old), text)), repo=target)
    cases.append({
        "name": "known-bad patch", "expected": "passed=False and the file is named",
        "ok": (not r.passed) and path in r.error,
        "detail": r.error or f"passed={r.passed} (no error text)",
    })

    ghost = "ghost_file_for_validation_check.py"
    r = pipeline.validate(_proposal(_diff(ghost, "a = 1\n", "a = 2\n")), repo=target)
    cases.append({
        "name": "file that does not exist", "expected": "passed=False, says it does not exist",
        "ok": (not r.passed) and "does not exist" in r.error,
        "detail": r.error or f"passed={r.passed} (no error text)",
    })

    r = pipeline.validate(
        _proposal(_diff(path, text, "".join(good_lines))),
        repo=RepoTarget(owner=owner, repo=NO_SUCH_REPO),
    )
    cases.append({
        "name": "repo that does not exist", "expected": "passed=False, clear error",
        "ok": (not r.passed) and "Could not get the repository files" in r.error,
        "detail": r.error or f"passed={r.passed} (no error text)",
    })
    return cases


def main(argv: list) -> int:
    owner = argv[1] if len(argv) > 1 else "psf"
    repo = argv[2] if len(argv) > 2 else "requests"
    path = argv[3] if len(argv) > 3 else "src/requests/sessions.py"

    from backend.github.repo_snapshot import GitHubSnapshotProvider, SnapshotError

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        try:
            from backend.config import get_settings

            token = get_settings().github_token.get_secret_value().strip()
        except Exception:
            token = ""
    print(f"Repo: {owner}/{repo}  file: {path}  token: {'yes' if token else 'no (60 requests/hour)'}")

    try:
        cases = run_cases(GitHubSnapshotProvider(token=token or None), owner, repo, path)
    except SnapshotError as exc:
        print(f"Setup problem: {exc}")
        return 2
    except RuntimeError as exc:
        print(f"Setup problem: {exc}")
        return 2

    for c in cases:
        print(f"\n[{'PASS' if c['ok'] else 'FAIL'}] {c['name']}  (expected: {c['expected']})")
        print(f"    {c['detail'][:300]}")
    good = sum(1 for c in cases if c["ok"])
    print(f"\n{good}/{len(cases)} behaved correctly")
    print("RESULT:", "PASS" if good == len(cases) else "FAIL")
    return 0 if good == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
