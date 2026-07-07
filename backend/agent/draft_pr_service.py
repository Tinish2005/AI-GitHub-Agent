"""
Draft PR service.

Turns a DraftPRRequest into a real (or fake) draft pull request:

    - `GitHubDraftPRService`: uses the GitHub REST API to create a
      branch, commit the patch, and open a DRAFT PR (never a mergeable
      PR - the human reviewer always has the final word).

    - `FakeDraftPRService`: returns deterministic fake results so
      tests never hit the network.

Design notes:
    - Validation gate: if `validation_passed=False` we refuse to create
      the PR and return a skipped_reason. This is the "autonomous-but-
      safe" contract from the project brief.
    - Branch names are goal-derived + timestamp so repeated runs don't
      collide.
    - Commit + PR are always DRAFT. Merging is a human decision.
    - GitHub commit-content creation requires a full file body, not a
      unified diff. Because our diff parsing is limited to
      new-file additions (Loop 12 sandbox), this service currently
      supports the same shape: it reads the '+' lines out of the diff
      to build the new file body. If the diff is anything more complex,
      we skip with a clear reason.
"""

from __future__ import annotations

import base64
import re
import time
from typing import Protocol

import httpx

from backend.agent.draft_pr_models import DraftPRRequest, DraftPRResult


GITHUB_API_BASE: str = "https://api.github.com"
DEFAULT_TIMEOUT_SECONDS: float = 20.0


class DraftPRService(Protocol):
    """Anything that can turn a DraftPRRequest into a DraftPRResult."""

    def create(self, request: DraftPRRequest) -> DraftPRResult:
        ...


# ---------------------------------------------------------------------------
# FakeDraftPRService: deterministic, used in tests
# ---------------------------------------------------------------------------


class FakeDraftPRService:
    """
    Deterministic fake that records requests and returns pre-canned results.

    Also enforces the same gates the real service would (validation must
    pass, diff must be non-empty), so tests can verify our safety rules.
    """

    def __init__(self, *, next_number: int = 42) -> None:
        self.calls: list = []
        self._next_number = next_number

    def create(self, request: DraftPRRequest) -> DraftPRResult:
        self.calls.append(request)

        if not request.validation_passed:
            return DraftPRResult(
                created=False,
                skipped_reason=(
                    f"Validation did not pass (score={request.validation_score:.2f}); "
                    "refusing to open a draft PR."
                ),
            )

        if not request.proposal_diff.strip():
            return DraftPRResult(
                created=False,
                error="Proposal diff is empty; nothing to publish.",
            )

        title = _build_title(request.goal)
        body = _build_body(request)
        branch = _build_branch(request.goal)
        number = self._next_number
        self._next_number += 1
        url = f"https://github.com/{request.owner}/{request.repo}/pull/{number}"

        return DraftPRResult(
            created=True,
            pr_number=number,
            pr_url=url,
            branch=branch,
            title=title,
            body=body,
        )


# ---------------------------------------------------------------------------
# GitHubDraftPRService: real client
# ---------------------------------------------------------------------------


class GitHubDraftPRService:
    """
    Real GitHub-backed draft PR creator.

    Requires a Personal Access Token with `repo` scope. Never posts a
    non-draft PR. Uses shallow HTTP calls with a fresh `httpx.Client`
    per public method - simple and predictable.
    """

    def __init__(
        self,
        token: str,
        *,
        base_url: str = GITHUB_API_BASE,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not token:
            raise ValueError("GitHub token must not be empty.")
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def create(self, request: DraftPRRequest) -> DraftPRResult:
        if not request.validation_passed:
            return DraftPRResult(
                created=False,
                skipped_reason=(
                    f"Validation did not pass (score={request.validation_score:.2f}); "
                    "refusing to open a draft PR."
                ),
            )
        if not request.proposal_diff.strip():
            return DraftPRResult(
                created=False,
                error="Proposal diff is empty; nothing to publish.",
            )

        file_ops = _extract_new_files_from_diff(request.proposal_diff)
        if not file_ops:
            return DraftPRResult(
                created=False,
                error=(
                    "Draft-PR publishing currently supports 'new file' diffs only. "
                    "Extend GitHubDraftPRService to handle modifications and deletions."
                ),
            )

        branch = _build_branch(request.goal)
        title = _build_title(request.goal)
        body = _build_body(request)

        try:
            base_sha = self._get_branch_sha(request.owner, request.repo, request.base_branch)
            self._create_branch(request.owner, request.repo, branch, base_sha)
            for path, content in file_ops.items():
                self._put_file(
                    request.owner, request.repo, branch, path, content,
                    f"[agent] {title}",
                )
            pr_url, pr_number = self._open_pr(
                request.owner, request.repo, branch, request.base_branch,
                title, body,
            )
        except Exception as exc:
            return DraftPRResult(
                created=False,
                error=f"GitHub API error: {type(exc).__name__}: {exc}",
                branch=branch,
                title=title,
                body=body,
            )

        return DraftPRResult(
            created=True,
            pr_number=pr_number,
            pr_url=pr_url,
            branch=branch,
            title=title,
            body=body,
        )

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
    ) -> dict:
        url = f"{self._base_url}{path}"
        with httpx.Client(timeout=self._timeout) as client:
            response = client.request(
                method, url, headers=self._headers(), json=json_body,
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"GitHub {response.status_code} on {method} {path}: {response.text[:200]}"
            )
        if response.text:
            return response.json()
        return {}

    def _get_branch_sha(self, owner: str, repo: str, branch: str) -> str:
        data = self._request("GET", f"/repos/{owner}/{repo}/git/ref/heads/{branch}")
        obj = data.get("object", {})
        sha = obj.get("sha", "")
        if not sha:
            raise RuntimeError(f"Could not resolve SHA for branch {branch}.")
        return str(sha)

    def _create_branch(self, owner: str, repo: str, new_branch: str, base_sha: str) -> None:
        self._request(
            "POST", f"/repos/{owner}/{repo}/git/refs",
            json_body={"ref": f"refs/heads/{new_branch}", "sha": base_sha},
        )

    def _put_file(
        self,
        owner: str, repo: str, branch: str,
        path: str, content: str, message: str,
    ) -> None:
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        self._request(
            "PUT", f"/repos/{owner}/{repo}/contents/{path.lstrip('/')}",
            json_body={
                "message": message,
                "content": encoded,
                "branch": branch,
            },
        )

    def _open_pr(
        self,
        owner: str, repo: str,
        head: str, base: str,
        title: str, body: str,
    ) -> tuple:
        data = self._request(
            "POST", f"/repos/{owner}/{repo}/pulls",
            json_body={
                "title": title, "body": body,
                "head": head, "base": base,
                "draft": True,
            },
        )
        url = str(data.get("html_url", ""))
        number = int(data.get("number", 0))
        return url, number


# ---------------------------------------------------------------------------
# Helpers used by both fake and real implementations
# ---------------------------------------------------------------------------


def _build_branch(goal: str) -> str:
    """Derive a stable-enough branch name from the goal + a timestamp."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", goal.strip().lower()).strip("-")
    if not slug:
        slug = "agent-fix"
    slug = slug[:40]
    stamp = int(time.time())
    return f"agent/{slug}-{stamp}"


def _build_title(goal: str) -> str:
    return f"[agent] {goal.strip()}"[:120]


def _build_body(request: DraftPRRequest) -> str:
    return (
        f"### Goal\n{request.goal}\n\n"
        f"### Confidence\n{request.confidence:.2f}\n\n"
        f"### Validation\n"
        f"Passed: {request.validation_passed}  |  Score: {request.validation_score:.2f}\n\n"
        f"### Explanation\n{request.proposal_explanation}\n\n"
        f"### Proposed diff\n```diff\n{request.proposal_diff}\n```\n\n"
        f"---\n"
        f"_This draft PR was opened by the AI GitHub Agent. Human review required before merge._"
    )


def _extract_new_files_from_diff(diff_text: str) -> dict:
    """
    Extract {path: file_body} for `--- /dev/null` (new-file) hunks.

    Supports the shape used by our validator's test suite. Anything more
    complex returns an empty dict so callers can refuse cleanly.
    """
    files: dict = {}
    current_path: str | None = None
    is_new_file = False
    body_lines: list = []

    for raw in diff_text.splitlines():
        if raw.startswith("--- "):
            # Reset state whenever we see a source header.
            current_path = None
            is_new_file = raw.startswith("--- /dev/null")
            body_lines = []
            continue
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            if target.startswith("b/"):
                target = target[2:]
            current_path = target or None
            continue
        if raw.startswith("@@"):
            continue
        if current_path is None or not is_new_file:
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            body_lines.append(raw[1:])
            files[current_path] = "\n".join(body_lines) + "\n"
    return files