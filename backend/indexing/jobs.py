"""Background indexing jobs for visitors.

A visitor asks to index a public GitHub repo. The server answers at once with a
job id, runs the (slow) indexing in a background thread, and the visitor checks
the job's status. Only one indexing job runs at a time, and every limit in
backend.limits applies. Jobs live in memory: they are lost when the server restarts.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field

from backend.github.models import RepoCoord
from backend.github.repo_check import RepoCheckError
from backend.indexing.vector_store import EmbeddingMismatchError
from backend.limits import LimitError

SECONDS_PER_CHUNK = 0.8   # embedding is paced at 0.7 s per chunk, plus overhead
KEEP_FINISHED_SECONDS = 3600
START_MESSAGE = (
    "Indexing takes a few minutes (usually 1 to 5). You can leave this page and check "
    "the status later with the job id."
)


def normalize_folder(path: str) -> str:
    """Turn what a visitor typed into a safe folder prefix like "src/mypackage/", or "" for the whole repo."""
    p = (path or "").strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.strip("/")
    if not p:
        return ""
    parts = p.split("/")
    if len(p) > 200 or any(part in ("", ".", "..") for part in parts) or not re.fullmatch(r"[A-Za-z0-9_./-]+", p):
        raise LimitError("The folder must be a path inside the repository, like src/mypackage.", status=400)
    return p + "/"


@dataclass
class IndexJob:
    id: str
    repo: str
    created_at: float
    status: str = "queued"          # queued, running, done, failed
    phase: str = "queued"           # queued, fetching, embedding, finished
    started_at: float | None = None
    finished_at: float | None = None
    files_done: int = 0
    files_total: int = 0
    chunks_total: int = 0
    files_indexed: int = 0
    chunks_indexed: int = 0
    files_skipped: int = 0
    truncated: bool = False
    message: str = START_MESSAGE
    error: str = ""
    _clock: object = field(default=time.time, repr=False)

    def view(self) -> dict:
        now = self._clock()
        started = self.started_at or self.created_at
        elapsed = int((self.finished_at or now) - started)
        estimate = int(self.chunks_total * SECONDS_PER_CHUNK) if self.chunks_total else None
        return {
            "job_id": self.id,
            "repo": self.repo,
            "status": self.status,
            "phase": self.phase,
            "files_done": self.files_done,
            "files_total": self.files_total,
            "chunks_total": self.chunks_total,
            "estimated_seconds": estimate,
            "elapsed_seconds": elapsed,
            "files_indexed": self.files_indexed,
            "chunks_indexed": self.chunks_indexed,
            "files_skipped": self.files_skipped,
            "truncated": self.truncated,
            "message": self.message,
            "error": self.error,
        }


class IndexJobManager:
    def __init__(
        self,
        *,
        guard,
        settings,
        store_factory,
        indexer_factory,
        repo_checker,
        run_in_thread: bool = True,
        clock=time.time,
        max_jobs: int = 100,
    ) -> None:
        self.guard = guard
        self.settings = settings
        self._store_factory = store_factory
        self._indexer_factory = indexer_factory
        self._repo_checker = repo_checker
        self._run_in_thread = run_in_thread
        self._clock = clock
        self._max_jobs = max_jobs
        self._jobs: dict = {}
        self._registry_lock = threading.Lock()

    # ------------------------------------------------------------------
    def get(self, job_id: str) -> dict | None:
        with self._registry_lock:
            job = self._jobs.get(job_id)
            return job.view() if job else None

    def submit(self, owner: str, repo: str, ip: str, is_admin: bool = False, path: str = "") -> dict:
        """Check the request, then start a job. Raises LimitError for every refusal."""
        s = self.settings
        if not (s.public_indexing_enabled or is_admin):
            raise LimitError("Indexing is closed on this server.", status=403)
        if not is_admin:
            self.guard.check_attempt(ip)

        folder = normalize_folder(path)

        try:
            info = self._repo_checker(owner, repo)
        except RepoCheckError as exc:
            raise LimitError(exc.message, status=exc.status) from exc
        label = info["full_name"]
        full_owner, _, full_repo = label.partition("/")

        existing = self._find_existing(label)
        if existing is not None:
            if existing["demo"]:
                raise LimitError(
                    "This is the built-in demo repository. It is already indexed.", status=409,
                )
            if not folder:  # naming a folder means "add this part", so it is allowed
                return {
                    "status": "already_indexed", "job_id": None, "repo": label,
                    "chunks_indexed": existing["chunks"],
                    "message": f"{label} is already indexed ({existing['chunks']} chunks).",
                }

        if not self.guard.index_lock.acquire(blocking=False):
            raise LimitError(
                "Another repository is being indexed. Try again in a few minutes.",
                status=429, retry_after=60,
            )
        try:
            if not is_admin:
                self.guard.charge_index_job(ip)
        except Exception:
            self.guard.index_lock.release()
            raise

        job = IndexJob(id=uuid.uuid4().hex, repo=label, created_at=self._clock(), _clock=self._clock)
        with self._registry_lock:
            self._forget_old_jobs()
            self._jobs[job.id] = job
        if self._run_in_thread:
            threading.Thread(
                target=self._run, args=(job, full_owner, full_repo, is_admin, folder), daemon=True,
            ).start()
        else:
            self._run(job, full_owner, full_repo, is_admin, folder)
        return job.view()

    # ------------------------------------------------------------------
    def _find_existing(self, label: str) -> dict | None:
        wanted = label.lower()
        for name, entry in self._store_factory().repo_summaries().items():
            if name.lower() == wanted:
                return entry
        return None

    def _forget_old_jobs(self) -> None:
        now = self._clock()
        for job_id in [
            j.id for j in self._jobs.values()
            if j.finished_at is not None and now - j.finished_at > KEEP_FINISHED_SECONDS
        ]:
            self._jobs.pop(job_id, None)
        while len(self._jobs) >= self._max_jobs:
            finished = [j for j in self._jobs.values() if j.finished_at is not None]
            if not finished:
                break
            self._jobs.pop(min(finished, key=lambda j: j.finished_at).id, None)

    def _make_room(self, store, needed: int) -> None:
        """Remove the oldest visitor repos until `needed` chunks fit. The demo is never removed."""
        cap = self.settings.public_index_store_max_chunks
        if store.count() + needed <= cap:
            return
        victims = sorted(
            (entry["indexed_at"], name)
            for name, entry in store.repo_summaries().items() if not entry["demo"]
        )
        for _, name in victims:
            store.remove_repo(name)
            if store.count() + needed <= cap:
                return
        raise LimitError("The server's index is full. Try again later.", status=503, retry_after=3600)

    def _run(self, job: IndexJob, owner: str, repo: str, is_admin: bool, folder: str = "") -> None:
        s = self.settings
        try:
            job.status, job.phase, job.started_at = "running", "fetching", self._clock()
            store = self._store_factory()
            indexer = self._indexer_factory(store, s.public_index_max_files)

            def progress(phase: str, done: int, total: int) -> None:
                job.phase = phase
                if phase == "fetching":
                    job.files_done, job.files_total = done, total
                else:
                    job.chunks_total = total

            def before_add(chunks: int) -> None:
                if not is_admin:
                    self.guard.check_embeddings(chunks)
                self._make_room(store, chunks)
                if not is_admin:
                    self.guard.charge_embeddings(chunks)

            extra = {"include_prefixes": (folder,)} if folder else {}  # only passed when asked for
            result = indexer.index(
                RepoCoord(owner=owner, repo=repo),
                max_chunks=s.public_index_max_chunks,
                max_file_bytes=s.public_index_max_file_bytes,
                progress=progress,
                before_add=before_add,
                **extra,
            )
            job.files_indexed = result.files_indexed
            job.chunks_indexed = result.chunks_indexed
            job.files_skipped = result.files_skipped
            job.truncated = result.truncated
            if result.chunks_indexed == 0:
                job.status = "failed"
                job.error = (
                    f"No Python files were found in the folder '{folder.rstrip('/')}'. Check the folder name."
                    if folder else
                    "No Python code was found to index in the first files of this repository."
                )
                job.message = job.error
            else:
                job.status, job.phase = "done", "finished"
                note = " Some files were skipped (too large, or over the limit)." if (result.truncated or result.files_skipped) else ""
                job.message = (
                    f"Indexed {result.files_indexed} files into {result.chunks_indexed} chunks.{note}"
                )
        except LimitError as exc:
            job.status, job.error = "failed", exc.message
            job.message = exc.message
        except EmbeddingMismatchError as exc:
            job.status, job.error = "failed", str(exc)
            job.message = "The server's search settings changed. Please tell the owner."
        except Exception as exc:
            job.status = "failed"
            job.error = f"Indexing failed ({type(exc).__name__}). Try again later."
            job.message = job.error
        finally:
            job.finished_at = self._clock()
            if job.status == "failed":
                job.phase = "finished"
            self.guard.index_lock.release()
