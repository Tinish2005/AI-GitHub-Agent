"""LIVE check for Loop 4b: open a real DRAFT pull request that changes an existing file.

This WRITES to GitHub. Run it only on a repository you own and do not mind
touching (a throw-away test repo is best). Run from the project root, in PowerShell:

    python scripts\\live_check_pr.py --owner YOUR_NAME --repo YOUR_TEST_REPO --yes
    python scripts\\live_check_pr.py --owner YOUR_NAME --repo YOUR_TEST_REPO --yes --cleanup

It adds ONE comment line to README.md (change it with --file), on a new branch
named agent/..., and opens a DRAFT pull request. It never touches your base
branch and never merges. Then it checks, using GitHub itself, that:
    1. the PR exists and is a draft
    2. the file on the new branch is exactly the old file plus the one line
    3. diffing the old file against the branch file gives back the reviewed diff
With --cleanup it closes the PR and deletes the branch afterwards.

Your GitHub token comes from GITHUB_TOKEN or your .env file (it needs write
access to the repo's contents and pull requests). It is never printed.
Exit code: 0 = all checks passed, 1 = a check failed, 2 = could not start.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _diff(path: str, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    )).rstrip("\n")


def _comment_line(path: str, stamp: str) -> str:
    if path.lower().endswith((".md", ".html", ".htm", ".xml")):
        return f"<!-- agent live check {stamp} -->"
    return f"# agent live check {stamp}"


def run_check(service, owner: str, repo: str, base_branch: str, path: str, *, cleanup: bool, now=None) -> list:
    """Do the whole check with a GitHubDraftPRService. Returns [(PASS|FAIL, message)]."""
    from backend.agent.draft_pr_models import DraftPRRequest

    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%SZ")
    results: list = []

    base_sha = service._get_branch_sha(owner, repo, base_branch)
    existing = service._get_file(owner, repo, path, base_sha)
    if existing is None:
        return [("FAIL", f"{path} does not exist on {base_branch}. Pass another file with --file.")]
    _, old_text = existing
    eol = "\r\n" if old_text.count("\r\n") > old_text.count("\n") - old_text.count("\r\n") else "\n"
    body = old_text if old_text.endswith(("\n", "\r")) or not old_text else old_text + eol
    new_text = body + _comment_line(path, stamp) + eol
    # A diff cannot describe a missing final newline without a special marker, so diff
    # against the same text with a final newline. The service keeps the original lines.
    old_for_diff = old_text.replace("\r\n", "\n")
    if old_for_diff and not old_for_diff.endswith("\n"):
        old_for_diff += "\n"
    diff = _diff(path, old_for_diff, new_text.replace("\r\n", "\n"))

    result = service.create(DraftPRRequest(
        owner=owner, repo=repo, base_branch=base_branch,
        goal=f"live check: add a comment line to {path}",
        proposal_explanation="Live check of the agent's pull request creation. Safe to close.",
        proposal_diff=diff, confidence=1.0, validation_passed=True, validation_score=1.0,
    ))
    if not result.created:
        return [("FAIL", f"The pull request was not created: {result.error or result.skipped_reason}")]
    results.append(("PASS", f"Draft PR #{result.pr_number} opened: {result.pr_url}"))

    pr = service._request("GET", f"/repos/{owner}/{repo}/pulls/{result.pr_number}")
    results.append(("PASS" if pr.get("draft") is True else "FAIL", f"The pull request is a draft: {pr.get('draft')}"))

    data = service._request(
        "GET", f"/repos/{owner}/{repo}/contents/{quote(path, safe='/')}?ref={quote(result.branch, safe='')}",
    )
    committed = base64.b64decode(data.get("content", "")).decode("utf-8")
    results.append((
        "PASS" if committed == new_text else "FAIL",
        "The file on the new branch is exactly the old file plus one line.",
    ))
    again = _diff(path, old_for_diff, committed.replace("\r\n", "\n"))
    results.append((
        "PASS" if again == diff else "FAIL",
        "Diffing the old file against the branch file gives back the reviewed diff.",
    ))
    base_now = service._get_branch_sha(owner, repo, base_branch)
    results.append((
        "PASS" if base_now == base_sha else "WARN",
        "The base branch was not changed." if base_now == base_sha
        else "The base branch moved while the check ran (someone else pushed). Not caused by this check.",
    ))

    if cleanup:
        try:
            service._request("PATCH", f"/repos/{owner}/{repo}/pulls/{result.pr_number}", json_body={"state": "closed"})
            gone = service._delete_branch_quietly(owner, repo, result.branch)
            results.append(("PASS" if gone else "WARN", "Closed the PR and deleted the branch." if gone
                            else f"Closed the PR, but could not delete branch {result.branch}."))
        except Exception as exc:
            results.append(("WARN", f"Clean-up failed ({type(exc).__name__}). Close the PR and delete {result.branch} by hand."))
    else:
        results.append(("INFO", f"Not cleaned up. Close PR #{result.pr_number} and delete branch {result.branch} when done."))
    return results


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="Live check: open a draft PR that changes a file.")
    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base", default="main", help="base branch (default main)")
    parser.add_argument("--file", default="README.md", help="existing text file to add a comment line to")
    parser.add_argument("--yes", action="store_true", help="confirm that this writes to GitHub")
    parser.add_argument("--cleanup", action="store_true", help="close the PR and delete the branch afterwards")
    args = parser.parse_args(argv[1:])

    if not args.yes:
        print(f"This will open a DRAFT pull request on {args.owner}/{args.repo} that adds one comment line to {args.file}.")
        print("It writes to GitHub. Add --yes to run it (and --cleanup to close the PR and delete the branch afterwards).")
        return 2

    import os

    from backend.agent.draft_pr_service import GitHubDraftPRService

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        try:
            from backend.config import get_settings

            token = get_settings().github_token.get_secret_value().strip()
        except Exception:
            token = ""
    if not token:
        print("No GitHub token found. Set GITHUB_TOKEN or put it in your .env file.")
        return 2

    service = GitHubDraftPRService(token=token)
    try:
        results = run_check(service, args.owner, args.repo, args.base, args.file, cleanup=args.cleanup)
    except Exception as exc:
        print(f"Could not run the check: {type(exc).__name__}: {exc}")
        print("A 403 or 404 usually means the token cannot write to this repo, or the repo or branch name is wrong.")
        return 2

    for level, message in results:
        print(f"[{level:4}] {message}")
    failed = any(level == "FAIL" for level, _ in results)
    print("\nRESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
