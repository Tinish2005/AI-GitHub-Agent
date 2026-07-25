"""
Repository cloner.

Wraps GitPython so we can clone any public GitHub repo into a local
cache directory. Uses SHA-256 hashing of the URL to derive a stable
folder name, so repeated clones of the same URL land in the same spot
(and can be skipped if already present).

A `Cloner` Protocol is exposed so any downstream code can accept either
the real `GitCloner` or the `FakeCloner` used by tests.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class CloneResult:
    """Return value for a completed clone."""

    url: str
    local_path: Path
    was_cached: bool


class Cloner(Protocol):
    """Anything that can clone a repo URL into a local directory."""

    def clone(self, url: str, *, force: bool = False) -> CloneResult:
        ...


def repo_folder_name(url: str) -> str:
    """Return a stable folder name derived from the repo URL."""
    if not url:
        raise ValueError("Repo URL must not be empty.")
    digest = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]
    tail = url.rstrip("/").split("/")[-1]
    tail = tail.removesuffix(".git") or "repo"
    return f"{tail}-{digest}"


class GitCloner:
    """
    Real cloner backed by GitPython.

    The cache directory is created lazily. GitPython is imported lazily
    inside `clone()` so importing this module in test environments does
    not require the `git` executable to be present.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def _target_for(self, url: str) -> Path:
        return self.cache_dir / repo_folder_name(url)

    def clone(self, url: str, *, force: bool = False) -> CloneResult:
        """Clone `url` into the cache. If already present, skip unless `force`."""
        if not url:
            raise ValueError("Repo URL must not be empty.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self._target_for(url)

        if target.exists():
            if not force:
                return CloneResult(url=url, local_path=target, was_cached=True)
            shutil.rmtree(target, ignore_errors=True)

        # Local import so pytest can collect tests even if git executable is missing.
        try:
            from git import Repo
        except ImportError as exc:
            raise RuntimeError(
                "GitPython is not installed. Run: pip install -r requirements.txt"
            ) from exc

        Repo.clone_from(url, str(target), depth=1)
        return CloneResult(url=url, local_path=target, was_cached=False)


class FakeCloner:
    """
    Deterministic cloner used in tests.

    Instead of hitting the network, it materializes a small synthetic
    Python project on disk (a couple of files) so the rest of the
    indexing pipeline can chunk, embed, and store the result.
    """

    DEFAULT_FILES: dict = {
        "app.py": (
            '"""Fake app module."""\n\n'
            "def hello(name: str) -> str:\n"
            '    """Return a greeting."""\n'
            '    return f"Hello, {name}"\n'
        ),
        "utils.py": (
            '"""Fake utils module."""\n\n'
            "def add(a: int, b: int) -> int:\n"
            '    """Add two numbers."""\n'
            "    return a + b\n"
        ),
    }

    def __init__(self, cache_dir: Path, files: dict | None = None) -> None:
        self.cache_dir = cache_dir
        self.files = files if files is not None else dict(self.DEFAULT_FILES)

    def clone(self, url: str, *, force: bool = False) -> CloneResult:
        if not url:
            raise ValueError("Repo URL must not be empty.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / repo_folder_name(url)

        if target.exists():
            if not force:
                return CloneResult(url=url, local_path=target, was_cached=True)
            shutil.rmtree(target, ignore_errors=True)

        target.mkdir(parents=True, exist_ok=True)
        for rel_path, content in self.files.items():
            file_path = target / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        return CloneResult(url=url, local_path=target, was_cached=False)