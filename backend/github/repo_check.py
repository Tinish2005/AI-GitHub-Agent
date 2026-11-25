"""Check that a GitHub repository is public before a visitor can index it.

The server's own GitHub token may be able to read private repositories, so
this check is what stops a visitor from indexing one through it.
"""

from __future__ import annotations

import httpx

from backend.github.repo_indexer import GITHUB_API_BASE, _make_ssl_context
from backend.github.repo_snapshot import NAME_PATTERN

NOT_PUBLIC_MESSAGE = "That repository was not found, or it is not a public repository."


class RepoCheckError(Exception):
    """The repository cannot be indexed. The message is safe to show."""

    def __init__(self, message: str, status: int = 404) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def check_public_repo(owner: str, repo: str, token: str | None = None, *, transport=None, timeout: float = 15.0) -> dict:
    """Return {"full_name", "default_branch"} for a public repo, or raise RepoCheckError.

    Missing and private repositories get the same answer, so nothing leaks about private repos.
    """
    if not NAME_PATTERN.match(owner or "") or not NAME_PATTERN.match(repo or ""):
        raise RepoCheckError("Invalid owner or repository name.", 422)
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        client = (
            httpx.Client(timeout=timeout, transport=transport)
            if transport is not None else httpx.Client(timeout=timeout, verify=_make_ssl_context())
        )
        with client:
            response = client.get(f"{GITHUB_API_BASE}/repos/{owner}/{repo}", headers=headers)
    except httpx.HTTPError as exc:
        raise RepoCheckError("Could not reach GitHub. Try again later.", 503) from exc

    if response.status_code == 404:
        raise RepoCheckError(NOT_PUBLIC_MESSAGE, 404)
    if response.status_code in (401, 403, 429):
        if response.headers.get("x-ratelimit-remaining") == "0" or "rate limit" in response.text.lower():
            raise RepoCheckError("GitHub is limiting requests right now. Try again later.", 503)
        raise RepoCheckError(NOT_PUBLIC_MESSAGE, 404)
    if response.status_code >= 400:
        raise RepoCheckError("GitHub returned an error. Try again later.", 503)

    data = response.json()
    if data.get("private") is True or str(data.get("visibility", "public")).lower() != "public":
        raise RepoCheckError(NOT_PUBLIC_MESSAGE, 404)
    return {
        "full_name": str(data.get("full_name") or f"{owner}/{repo}"),
        "default_branch": str(data.get("default_branch") or "main"),
    }
