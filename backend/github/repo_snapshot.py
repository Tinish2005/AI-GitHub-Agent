"""Fetch the real files a patch touches from GitHub, at one exact commit.

Validation needs the real text of the files a patch changes. This module asks
the GitHub REST API for:
    1. the exact commit (a branch or tag is turned into a commit SHA),
    2. the list of file paths in the repo (so imports can be checked),
    3. the content of only the files the patch touches.

Nothing is written to disk here. The caller decides what to do with the text.
Public repos work without a token. Private repos need a token that can read them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

import httpx

from backend.github.repo_indexer import GITHUB_API_BASE, _make_ssl_context


NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
REF_PATTERN = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")
MAX_FILE_BYTES = 1_000_000
MAX_FILES = 20
MAX_TREE_PATHS = 5000


class SnapshotError(Exception):
    """The repository files could not be fetched. The message is safe to show."""


class SnapshotNotFound(SnapshotError):
    """GitHub answered 404."""


@dataclass(frozen=True)
class RepoSnapshot:
    """What was fetched: the commit, the file texts, and the repo's .py paths."""

    commit: str
    files: dict
    missing: tuple = ()
    tree_paths: tuple = ()


class SnapshotProvider(Protocol):
    """Anything that can fetch a RepoSnapshot. Tests use a fake one."""

    def fetch(self, owner: str, repo: str, ref: str | None, paths: list) -> RepoSnapshot:
        ...


def safe_repo_path(path: str) -> str:
    """Return a clean relative POSIX path, or raise SnapshotError for unsafe ones."""
    raw = str(path or "").strip()
    if not raw or "\x00" in raw or "\\" in raw:
        raise SnapshotError(f"Unsafe file path in the patch: {raw[:80]!r}")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise SnapshotError(f"Unsafe file path in the patch (absolute path): {raw[:80]!r}")
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise SnapshotError(f"Unsafe file path in the patch (outside the repo): {raw[:80]!r}")
    if parts[0] == ".git":
        raise SnapshotError(f"Unsafe file path in the patch (.git folder): {raw[:80]!r}")
    return "/".join(parts)


class GitHubSnapshotProvider:
    """Real provider backed by the GitHub REST API."""

    def __init__(
        self,
        token: str | None = None,
        *,
        timeout: float = 20.0,
        max_file_bytes: int = MAX_FILE_BYTES,
        max_files: int = MAX_FILES,
        max_tree_paths: int = MAX_TREE_PATHS,
        transport=None,
    ) -> None:
        self._token = token or None
        self._timeout = timeout
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self.max_tree_paths = max_tree_paths
        self._transport = transport
        self._verify = _make_ssl_context()

    # ------------------------------------------------------------------
    def fetch(self, owner: str, repo: str, ref: str | None, paths: list) -> RepoSnapshot:
        if not NAME_PATTERN.match(owner or "") or not NAME_PATTERN.match(repo or ""):
            raise SnapshotError("Invalid owner or repo name.")
        ref_text = (ref or "").strip() or "HEAD"
        if not REF_PATTERN.match(ref_text) or ".." in ref_text:
            raise SnapshotError("Invalid ref (branch, tag or commit).")

        wanted = list(dict.fromkeys(safe_repo_path(p) for p in paths))
        if len(wanted) > self.max_files:
            raise SnapshotError(
                f"The patch changes {len(wanted)} files; the limit is {self.max_files}."
            )

        with self._client() as client:
            base = f"/repos/{owner}/{repo}"
            commit = self._get(client, f"{base}/commits/{quote(ref_text, safe='/')}").get("sha", "")
            if not commit:
                raise SnapshotError("GitHub did not return a commit for that ref.")
            tree = self._get(client, f"{base}/git/trees/{commit}", params={"recursive": "1"})
            blobs = {
                n.get("path", ""): int(n.get("size") or 0)
                for n in tree.get("tree", [])
                if n.get("type") == "blob"
            }
            truncated = bool(tree.get("truncated"))
            tree_paths = tuple(sorted(p for p in blobs if p.endswith(".py")))[: self.max_tree_paths]

            files: dict = {}
            missing: list = []
            for path in wanted:
                if path not in blobs and not truncated:
                    missing.append(path)
                    continue
                if blobs.get(path, 0) > self.max_file_bytes:
                    raise SnapshotError(
                        f"{path} is too large ({blobs[path]} bytes; the limit is {self.max_file_bytes})."
                    )
                try:
                    raw = self._get(
                        client, f"{base}/contents/{quote(path, safe='/')}",
                        params={"ref": commit}, raw=True,
                    )
                except SnapshotNotFound:
                    missing.append(path)
                    continue
                if len(raw) > self.max_file_bytes:
                    raise SnapshotError(
                        f"{path} is too large ({len(raw)} bytes; the limit is {self.max_file_bytes})."
                    )
                try:
                    files[path] = raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise SnapshotError(f"{path} is not a UTF-8 text file.") from exc

        return RepoSnapshot(
            commit=commit, files=files, missing=tuple(missing), tree_paths=tree_paths,
        )

    # ------------------------------------------------------------------
    def _client(self) -> httpx.Client:
        if self._transport is not None:
            return httpx.Client(timeout=self._timeout, transport=self._transport)
        return httpx.Client(timeout=self._timeout, verify=self._verify)

    def _headers(self, raw: bool) -> dict:
        accept = "application/vnd.github.raw+json" if raw else "application/vnd.github+json"
        headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _get(self, client: httpx.Client, path: str, *, params: dict | None = None, raw: bool = False):
        try:
            r = client.get(f"{GITHUB_API_BASE}{path}", headers=self._headers(raw), params=params)
        except httpx.HTTPError as exc:
            raise SnapshotError(f"Could not reach GitHub ({type(exc).__name__}).") from exc
        if r.status_code == 404:
            raise SnapshotNotFound(
                f"Not found on GitHub: {path}. Check the owner, repo and ref. "
                "Private repos need a token that can read them."
            )
        if r.status_code == 401:
            raise SnapshotError("GitHub rejected the token.")
        if r.status_code in (403, 429):
            if r.headers.get("x-ratelimit-remaining") == "0" or "rate limit" in r.text.lower():
                raise SnapshotError(
                    "GitHub rate limit reached. Try again later, or set a GitHub token."
                )
            raise SnapshotError("GitHub refused access (403).")
        if r.status_code >= 400:
            raise SnapshotError(f"GitHub error {r.status_code}.")
        return r.content if raw else r.json()
