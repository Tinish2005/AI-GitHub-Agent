"""
Indexing service.

Orchestrates the full offline pipeline for a single repository:
    1. Clone the repo (via a Cloner) into the local cache.
    2. Walk the cloned directory and parse every .py file into chunks.
    3. Extract metadata for each chunk (hashes + imports + calls).
    4. Store chunks + metadata + embeddings in the vector store.

The service is transport-agnostic: give it a Cloner and a VectorStore
and it will index. Tests inject a FakeCloner + fake embeddings so no
network is needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.cloning.cloner import Cloner, CloneResult
from backend.indexing.ast_parser import parse_directory
from backend.indexing.metadata import extract_metadata_for_chunks
from backend.indexing.vector_store import VectorStore


def repo_label_from_url(url: str) -> str:
    """Turn a clone URL into "owner/name" (best effort), e.g. for the stats page."""
    text = re.sub(r"\.git$", "", (url or "").strip().rstrip("/"))
    parts = [p for p in re.split(r"[/:]", text) if p]
    return "/".join(parts[-2:]) if len(parts) >= 2 else text


@dataclass(frozen=True)
class IndexResult:
    """Return value for a completed indexing run."""

    url: str
    local_path: Path
    was_cached: bool
    files_scanned: int
    chunks_indexed: int


class IndexingService:
    """End-to-end offline indexer: clone -> parse -> metadata -> vector store."""

    def __init__(
        self,
        cloner: Cloner,
        vector_store: VectorStore,
    ) -> None:
        self.cloner = cloner
        self.vector_store = vector_store

    def index_repo(self, url: str, *, force: bool = False) -> IndexResult:
        """Clone the URL if needed, then index every Python file."""
        if not url or not url.strip():
            raise ValueError("Repo URL must not be empty.")

        clone: CloneResult = self.cloner.clone(url, force=force)
        chunks = parse_directory(clone.local_path)

        files_scanned = self._count_python_files(clone.local_path)

        if not chunks:
            return IndexResult(
                url=clone.url,
                local_path=clone.local_path,
                was_cached=clone.was_cached,
                files_scanned=files_scanned,
                chunks_indexed=0,
            )

        metas = extract_metadata_for_chunks(chunks, project_root=clone.local_path)
        self.vector_store.add_chunks(
            chunks,
            metas,
            extra_metadata={
                "repo": repo_label_from_url(clone.url),
                "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                # Lets the stats page show paths relative to the clone folder.
                "root": str(clone.local_path),
            },
        )

        return IndexResult(
            url=clone.url,
            local_path=clone.local_path,
            was_cached=clone.was_cached,
            files_scanned=files_scanned,
            chunks_indexed=len(chunks),
        )

    @staticmethod
    def _count_python_files(root: Path) -> int:
        """Count .py files under root (mirrors parse_directory's exclusions)."""
        excluded = {".venv", "venv", "__pycache__", ".git", "node_modules"}
        count = 0
        for path in root.rglob("*.py"):
            if any(part in excluded for part in path.parts):
                continue
            count += 1
        return count
