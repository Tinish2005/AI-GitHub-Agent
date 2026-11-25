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
      unified diff. The real service therefore reads each file the diff
      changes from the base commit, applies the reviewed diff to it with
      the same strict rules as the validation step, and commits the
      resulting text. It supports new files, changes to existing files and
      deletions. All new content is computed BEFORE any branch is created,
      so a patch that does not fit leaves nothing behind on GitHub.
    - Never touched: renames, binary files, workflow files under
      .github/workflows/, files over 1 MB, and diffs over 20 files.
"""

from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

import httpx

from backend.agent.draft_pr_models import DraftPRRequest, DraftPRResult
from backend.agent.patching import apply_patch_to_text
from backend.agent.validation_pipeline import PatchApplyError, ValidationPipeline
from backend.github.repo_indexer import _make_ssl_context
from backend.github.repo_snapshot import MAX_FILE_BYTES, MAX_FILES, SnapshotError, safe_repo_path


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


class GitHubHTTPError(RuntimeError):
    """GitHub answered with an error status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class PRChangeError(Exception):
    """The diff cannot be turned into commits. The message is safe to show."""


@dataclass(frozen=True)
class ChangeOp:
    """One file the diff touches."""

    kind: str  # "add", "modify" or "delete"
    path: str
    patched_file: object


@dataclass(frozen=True)
class PreparedChange:
    """A file change ready to commit: the new content and the blob it replaces."""

    kind: str
    path: str
    content: str | None
    blob_sha: str | None


def _parse_changes(diff_text: str) -> list:
    """Split a diff into safe per-file operations, or raise PRChangeError."""
    try:
        from unidiff import PatchSet
    except ImportError as exc:  # pragma: no cover - unidiff is a requirement
        raise PRChangeError("unidiff is not installed on the server.") from exc
    duplicate = _first_duplicate_path(diff_text)
    if duplicate:
        # Checked on the raw text: old unidiff versions silently merge the same file twice.
        raise PRChangeError(f"{duplicate} appears twice in the diff.")
    try:
        patch = PatchSet(diff_text)
    except Exception as exc:
        raise PRChangeError(
            f"The diff could not be read ({type(exc).__name__}).{_old_parser_hint()}"
        ) from exc
    if len(patch) == 0:
        raise PRChangeError("The diff contains no file changes.")
    if len(patch) > MAX_FILES:
        raise PRChangeError(f"The diff changes {len(patch)} files; the limit is {MAX_FILES}.")

    operations: list = []
    seen: set = set()
    for patched_file in patch:
        if getattr(patched_file, "is_binary_file", False):
            raise PRChangeError("Binary file changes are not supported.")
        source = str(patched_file.source_file or "")
        target = str(patched_file.target_file or "")
        if (
            source != "/dev/null" and target != "/dev/null"
            and ValidationPipeline._normalized_path(patched_file) != _strip_prefix(source)
        ):
            raise PRChangeError("File renames are not supported.")
        try:
            path = safe_repo_path(ValidationPipeline._normalized_path(patched_file))
        except SnapshotError as exc:
            raise PRChangeError(str(exc)) from exc
        if path.startswith(".github/workflows/"):
            raise PRChangeError(f"{path} is a workflow file. The agent never changes workflow files.")
        if path in seen:
            raise PRChangeError(f"{path} appears twice in the diff.")
        seen.add(path)
        if patched_file.is_added_file:
            kind = "add"
        elif patched_file.is_removed_file:
            kind = "delete"
        else:
            kind = "modify"
        operations.append(ChangeOp(kind, path, patched_file))
    return operations


def _header_path(text: str) -> str:
    """The file name in a '--- ' or '+++ ' header line (no a/ b/ prefix, no timestamp)."""
    return _strip_prefix(text.split("\t")[0].strip())


def _first_duplicate_path(diff_text: str) -> str | None:
    """The first file that has two sections in the diff, or None."""
    lines = diff_text.splitlines()
    seen: set = set()
    for i in range(len(lines) - 2):
        # A real file header pair is followed by a hunk header; a changed line that
        # merely looks like one is followed by more diff content.
        if lines[i].startswith("--- ") and lines[i + 1].startswith("+++ ") and lines[i + 2].startswith("@@"):
            source, target = _header_path(lines[i][4:]), _header_path(lines[i + 1][4:])
            path = target if target != "/dev/null" else source
            if path in seen:
                return path
            seen.add(path)
    return None


def _old_parser_hint() -> str:
    """A hint when the diff reader (unidiff) is too old to read some diffs."""
    try:
        from importlib.metadata import version

        installed = version("unidiff")
        if int(installed.split(".")[0]) < 1:
            return f" The server's unidiff library is old ({installed}); install unidiff>=1.0.0."
    except Exception:
        pass
    return ""


def _strip_prefix(path: str) -> str:
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def _apply(op: ChangeOp, text: str) -> str:
    try:
        return apply_patch_to_text(text, op.patched_file, op.path)
    except PatchApplyError as exc:
        raise PRChangeError(f"The diff does not fit the repository: {exc}") from exc


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
        transport=None,
    ) -> None:
        if not token:
            raise ValueError("GitHub token must not be empty.")
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport

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

        branch = _build_branch(request.goal)
        title = _build_title(request.goal)
        body = _build_body(request)

        # 1. Work out every new file content BEFORE anything is created on GitHub.
        try:
            operations = _parse_changes(request.proposal_diff)
            base_sha = self._get_branch_sha(request.owner, request.repo, request.base_branch)
            prepared = self._prepare_changes(request.owner, request.repo, base_sha, operations)
        except PRChangeError as exc:
            return DraftPRResult(created=False, error=str(exc))
        except Exception as exc:
            return DraftPRResult(
                created=False,
                error=f"GitHub API error: {type(exc).__name__}: {exc}",
            )

        # 2. Branch from the same commit the files were read from, commit, open the draft PR.
        branch_created = False
        try:
            self._create_branch(request.owner, request.repo, branch, base_sha)
            branch_created = True
            message = f"[agent] {title}"
            for change in prepared:
                if change.kind == "delete":
                    self._delete_file(
                        request.owner, request.repo, branch, change.path, message, change.blob_sha,
                    )
                else:
                    self._put_file(
                        request.owner, request.repo, branch, change.path, change.content,
                        message, sha=change.blob_sha,
                    )
            pr_url, pr_number = self._open_pr(
                request.owner, request.repo, branch, request.base_branch,
                title, body,
            )
        except Exception as exc:
            error = f"GitHub API error: {type(exc).__name__}: {exc}"
            if branch_created:
                if self._delete_branch_quietly(request.owner, request.repo, branch):
                    error += f" (the branch {branch} was removed)"
                else:
                    error += f" (the branch {branch} could not be removed; delete it by hand)"
            return DraftPRResult(
                created=False,
                error=error,
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
        if self._transport is not None:
            client = httpx.Client(timeout=self._timeout, transport=self._transport)
        else:
            client = httpx.Client(timeout=self._timeout, verify=_make_ssl_context())
        with client:
            response = client.request(
                method, url, headers=self._headers(), json=json_body,
            )
        if response.status_code >= 400:
            raise GitHubHTTPError(
                response.status_code,
                f"GitHub {response.status_code} on {method} {path}: {response.text[:200]}",
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
        *,
        sha: str | None = None,
    ) -> None:
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        payload = {
            "message": message,
            "content": encoded,
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha  # required by GitHub when a file already exists
        self._request(
            "PUT", f"/repos/{owner}/{repo}/contents/{quote(path.lstrip('/'), safe='/')}",
            json_body=payload,
        )

    def _delete_file(
        self,
        owner: str, repo: str, branch: str,
        path: str, message: str, sha: str,
    ) -> None:
        self._request(
            "DELETE", f"/repos/{owner}/{repo}/contents/{quote(path.lstrip('/'), safe='/')}",
            json_body={"message": message, "sha": sha, "branch": branch},
        )

    def _delete_branch_quietly(self, owner: str, repo: str, branch: str) -> bool:
        """Best effort clean-up of a branch we made. Returns True when it is gone."""
        try:
            self._request("DELETE", f"/repos/{owner}/{repo}/git/refs/heads/{branch}")
            return True
        except Exception:
            return False

    def _get_file(self, owner: str, repo: str, path: str, ref: str):
        """Return (blob_sha, text) of a file at a commit, or None if it does not exist."""
        try:
            data = self._request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{quote(path, safe='/')}?ref={quote(ref, safe='')}",
            )
        except GitHubHTTPError as exc:
            if exc.status == 404:
                return None
            raise
        if not isinstance(data, dict) or data.get("type") != "file":
            raise PRChangeError(f"{path} is not a regular file in the repository.")
        if int(data.get("size") or 0) > MAX_FILE_BYTES:
            raise PRChangeError(f"{path} is too large to change here (over {MAX_FILE_BYTES} bytes).")
        encoded = data.get("content") or ""
        if data.get("encoding") != "base64" or (not encoded and int(data.get("size") or 0) > 0):
            raise PRChangeError(f"GitHub did not return the content of {path}.")
        try:
            text = base64.b64decode(encoded).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise PRChangeError(f"{path} is not a UTF-8 text file.") from exc
        return str(data.get("sha", "")), text

    def _prepare_changes(self, owner: str, repo: str, base_sha: str, operations: list) -> list:
        """Read each file at `base_sha` and compute its new content. Creates nothing."""
        prepared: list = []
        for op in operations:
            existing = self._get_file(owner, repo, op.path, base_sha)
            if op.kind == "add":
                if existing is not None:
                    raise PRChangeError(
                        f"The diff creates {op.path}, but that file already exists in the repository."
                    )
                content = _apply(op, "")
                prepared.append(PreparedChange("add", op.path, content, None))
                continue
            if existing is None:
                verb = "deletes" if op.kind == "delete" else "changes"
                raise PRChangeError(
                    f"The diff {verb} {op.path}, but that file does not exist in the repository."
                )
            blob_sha, text = existing
            new_text = _apply(op, text)
            if op.kind == "delete":
                if new_text.strip():
                    raise PRChangeError(
                        f"The diff deletes {op.path}, but its removed lines do not cover the whole file."
                    )
                prepared.append(PreparedChange("delete", op.path, None, blob_sha))
            else:
                prepared.append(PreparedChange("modify", op.path, new_text, blob_sha))
        return prepared

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