"""
GitHub repository indexer using the REST API (no Git required).

Fetches a repo''s Python files over HTTPS via the GitHub API, parses
them into chunks, and stores them in the vector store. Uses the OS
certificate store (via truststore) so it works behind corporate SSL
inspection proxies.
"""

from __future__ import annotations

import base64
import ssl
from dataclasses import dataclass
from pathlib import Path

import httpx

from backend.github.models import RepoCoord
from backend.indexing.ast_parser import parse_source
from backend.indexing.metadata import extract_metadata_for_chunks
from backend.indexing.vector_store import VectorStore


GITHUB_API_BASE = "https://api.github.com"
EXCLUDED_PARTS = {".venv", "venv", "__pycache__", ".git", "node_modules", "tests", "test"}


def _make_ssl_context():
    """Build an SSL context that trusts the OS certificate store."""
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        return True  # fall back to default verification


@dataclass(frozen=True)
class GitHubIndexResult:
    owner: str
    repo: str
    ref: str
    files_indexed: int
    chunks_indexed: int


class GitHubRepoIndexer:
    """Indexes a GitHub repo''s Python files via the REST API."""

    def __init__(self, token, vector_store, *, max_files=15, timeout=20.0):
        self._token = token or None
        self.vector_store = vector_store
        self.max_files = max_files
        self._timeout = timeout
        self._verify = _make_ssl_context()

    def _headers(self):
        h = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _get(self, path, params=None):
        url = f"{GITHUB_API_BASE}{path}"
        with httpx.Client(timeout=self._timeout, verify=self._verify) as client:
            r = client.get(url, headers=self._headers(), params=params)
        if r.status_code == 404:
            raise FileNotFoundError(f"GitHub 404: {path}")
        if r.status_code >= 400:
            raise RuntimeError(f"GitHub {r.status_code}: {r.text[:200]}")
        return r.json()

    def _default_branch(self, owner, repo):
        data = self._get(f"/repos/{owner}/{repo}")
        return data.get("default_branch", "main")

    def _list_python_files(self, owner, repo, ref):
        data = self._get(
            f"/repos/{owner}/{repo}/git/trees/{ref}",
            params={"recursive": "1"},
        )
        out = []
        for node in data.get("tree", []):
            if node.get("type") != "blob":
                continue
            path = node.get("path", "")
            if not path.endswith(".py"):
                continue
            if any(part in EXCLUDED_PARTS for part in path.split("/")):
                continue
            out.append(path)
        return out

    def _fetch_file(self, owner, repo, path, ref):
        data = self._get(
            f"/repos/{owner}/{repo}/contents/{path}",
            params={"ref": ref},
        )
        raw = data.get("content", "") or ""
        if data.get("encoding") == "base64":
            return base64.b64decode(raw).decode("utf-8", errors="replace")
        return raw

    def index(self, coord, ref=None):
        owner, repo = coord.owner, coord.repo
        branch = ref or self._default_branch(owner, repo)

        files = self._list_python_files(owner, repo, branch)
        files = files[: self.max_files]

        all_chunks = []
        for path in files:
            try:
                content = self._fetch_file(owner, repo, path, branch)
            except Exception:
                continue
            all_chunks.extend(parse_source(content, Path(path)))

        if not all_chunks:
            return GitHubIndexResult(owner, repo, branch, len(files), 0)

        metas = extract_metadata_for_chunks(all_chunks)
        self.vector_store.add_chunks(all_chunks, metas)
        return GitHubIndexResult(owner, repo, branch, len(files), len(all_chunks))
