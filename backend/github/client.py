"""
GitHub REST API client.

A typed wrapper around a small set of read-only GitHub operations:
fetch a file, list issues, get a pull request. Designed so any code
that depends on it can take either the real `GitHubClient` or the
`FakeGitHubClient` for tests - they share the same Protocol.
"""

from __future__ import annotations

import base64
from typing import Protocol

import httpx

from backend.github.models import GitHubFile, Issue, PullRequest, RepoCoord


GITHUB_API_BASE: str = "https://api.github.com"
DEFAULT_TIMEOUT_SECONDS: float = 15.0


class GitHubAPI(Protocol):
    """Minimal interface every GitHub backend must satisfy."""

    def get_file(self, coord: RepoCoord, path: str, *, ref: str | None = None) -> GitHubFile:
        ...

    def list_issues(self, coord: RepoCoord, *, state: str = "open", per_page: int = 30) -> list:
        ...

    def get_pr(self, coord: RepoCoord, number: int) -> PullRequest:
        ...


class FakeGitHubClient:
    """
    Deterministic fake used in tests.

    Holds a small in-memory snapshot of files / issues / PRs so tests
    can assert end-to-end behavior without ever touching api.github.com.
    """

    def __init__(self) -> None:
        self._files: dict = {}    # (owner, repo, path, ref) -> GitHubFile
        self._issues: dict = {}   # (owner, repo) -> list of Issue
        self._prs: dict = {}      # (owner, repo, number) -> PullRequest

    # --- Setup helpers used by tests ---
    def add_file(self, coord: RepoCoord, path: str, content: str, *, ref: str | None = None) -> None:
        f = GitHubFile(
            path=path,
            sha=f"fake-sha-{len(content)}",
            size=len(content.encode("utf-8")),
            content=content,
            encoding="utf-8",
        )
        self._files[(coord.owner, coord.repo, path, ref)] = f

    def add_issue(self, coord: RepoCoord, issue: Issue) -> None:
        self._issues.setdefault((coord.owner, coord.repo), []).append(issue)

    def add_pr(self, coord: RepoCoord, pr: PullRequest) -> None:
        self._prs[(coord.owner, coord.repo, pr.number)] = pr

    # --- Read API ---
    def get_file(self, coord: RepoCoord, path: str, *, ref: str | None = None) -> GitHubFile:
        key = (coord.owner, coord.repo, path, ref)
        if key not in self._files:
            raise FileNotFoundError(f"{coord.slug()}:{path} (ref={ref}) not found")
        return self._files[key]

    def list_issues(self, coord: RepoCoord, *, state: str = "open", per_page: int = 30) -> list:
        all_issues = self._issues.get((coord.owner, coord.repo), [])
        if state == "all":
            filtered = list(all_issues)
        else:
            filtered = [i for i in all_issues if i.state == state]
        return filtered[:per_page]

    def get_pr(self, coord: RepoCoord, number: int) -> PullRequest:
        key = (coord.owner, coord.repo, number)
        if key not in self._prs:
            raise FileNotFoundError(f"{coord.slug()}#{number} not found")
        return self._prs[key]

        

class GitHubClient:
    """
    Real GitHub REST API client.

    Uses a short-lived `httpx.Client` per call (cheap; the alternative is
    keeping a long-lived connection pool which is overkill here).
    Authentication is optional: when no token is given the client falls
    back to unauthenticated requests (60 req/hour rate limit).
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        base_url: str = GITHUB_API_BASE,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._token = token or None
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict:
        out = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            out["Authorization"] = f"Bearer {self._token}"
        return out

    def _request(self, method: str, path: str, *, params: dict | None = None) -> dict:
        """Issue an HTTP request and return parsed JSON. Raises on HTTP errors."""
        url = f"{self._base_url}{path}"
        with httpx.Client(timeout=self._timeout) as client:
            response = client.request(method, url, headers=self._headers(), params=params)
        if response.status_code == 404:
            raise FileNotFoundError(f"GitHub 404: {method} {path}")
        if response.status_code >= 400:
            raise RuntimeError(
                f"GitHub {response.status_code} on {method} {path}: {response.text[:200]}"
            )
        return response.json()

        

    def get_file(self, coord: RepoCoord, path: str, *, ref: str | None = None) -> GitHubFile:
        """Fetch a single file from a repo. Optional `ref` is a branch / tag / SHA."""
        if not path:
            raise ValueError("path must not be empty.")
        params = {"ref": ref} if ref else None
        data = self._request(
            "GET",
            f"/repos/{coord.owner}/{coord.repo}/contents/{path.lstrip('/')}",
            params=params,
        )
        if isinstance(data, list):
            raise ValueError(f"{path} is a directory, not a file.")
        raw_content = data.get("content", "") or ""
        encoding = data.get("encoding", "base64")
        if encoding == "base64":
            decoded = base64.b64decode(raw_content).decode("utf-8", errors="replace")
        else:
            decoded = raw_content
        return GitHubFile(
            path=data.get("path", path),
            sha=data.get("sha", ""),
            size=int(data.get("size", len(decoded))),
            content=decoded,
            encoding="utf-8",
        )

    def list_issues(self, coord: RepoCoord, *, state: str = "open", per_page: int = 30) -> list:
        """List issues for a repo. `state` is 'open', 'closed', or 'all'."""
        if state not in ("open", "closed", "all"):
            raise ValueError("state must be one of: open, closed, all")
        if per_page < 1 or per_page > 100:
            raise ValueError("per_page must be between 1 and 100")
        data = self._request(
            "GET",
            f"/repos/{coord.owner}/{coord.repo}/issues",
            params={"state": state, "per_page": per_page},
        )
        # GitHub returns PRs in the issues list too - filter them out.
        issues = []
        for item in data:
            if "pull_request" in item:
                continue
            issues.append(
                Issue(
                    number=item["number"],
                    title=item["title"],
                    state=item["state"],
                    author=(item.get("user") or {}).get("login", "unknown"),
                    body=item.get("body") or "",
                    url=item.get("html_url", ""),
                )
            )
        return issues

    def get_pr(self, coord: RepoCoord, number: int) -> PullRequest:
        """Fetch a single pull request by its number."""
        if number < 1:
            raise ValueError("PR number must be >= 1.")
        data = self._request(
            "GET",
            f"/repos/{coord.owner}/{coord.repo}/pulls/{number}",
        )
        head_label = (data.get("head") or {}).get("ref", "")
        base_label = (data.get("base") or {}).get("ref", "")
        return PullRequest(
            number=data["number"],
            title=data["title"],
            state=data["state"],
            author=(data.get("user") or {}).get("login", "unknown"),
            head=head_label or "unknown",
            base=base_label or "unknown",
            body=data.get("body") or "",
            url=data.get("html_url", ""),
            merged=bool(data.get("merged", False)),
        )