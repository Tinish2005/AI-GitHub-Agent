"""Step executors for the ExecutionEngine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from backend.agent.models import PlanStep, StepKind
from backend.agent.repo_context import format_chunks, retrieve_repo_context
from backend.github.client import GitHubAPI
from backend.github.models import RepoCoord
from backend.indexing.vector_store import VectorStore
from backend.rag.pipeline import RAGPipeline
from backend.agent.fix_generator import FixGenerator
from backend.agent.validation_pipeline import ValidationPipeline
from backend.agent.validation_models import ValidationResult
from backend.agent.draft_pr_models import DraftPRRequest, DraftPRResult
from backend.agent.draft_pr_service import DraftPRService


# Keys used to hand results from one step to the next (stored in prior_outputs).
FIX_PROPOSAL_KEY = "fix_proposal"
VALIDATION_RESULT_KEY = "validation_result"
# Full text of the retrieved code chunks (set by the RETRIEVE step).
REPO_CONTEXT_KEY = "repo_context"
# Which GitHub repo the VALIDATE step should check the patch against (a RepoTarget).
REPO_TARGET_KEY = "repo_target"

# Line that introduces the diff inside the GENERATE step's output.
PATCH_MARKER = "Patch (unified diff):"


def _repo_scope(prior_outputs: dict) -> str | None:
    """"owner/name" of the repo this run is about, if the request named one."""
    target = prior_outputs.get(REPO_TARGET_KEY)
    if target is None:
        return None
    return f"{target.owner}/{target.repo}"


class StepSkipped(Exception):
    """Raised by an executor that cannot do its job.

    Use it when an input is missing or the step is not configured. The
    engine records the step as SKIPPED and shows the message as the reason.
    A skipped step is not a failure, so it is never retried and it does
    not abort the run.
    """


@dataclass(frozen=True)
class StepContext:
    """Everything a step executor might need to do its job."""

    goal: str
    step: PlanStep
    prior_outputs: dict


class StepExecutor(Protocol):
    """Runs a single PlanStep and returns a plain string result."""

    kind: StepKind

    def run(self, context: StepContext) -> str:
        ...


class RetrieveExecutor:
    """Runs a vector search using the goal as the query."""

    kind: StepKind = StepKind.RETRIEVE

    def __init__(self, vector_store: VectorStore, *, top_k: int = 5) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        self.vector_store = vector_store
        self.top_k = top_k

    def run(self, context: StepContext) -> str:
        scope = _repo_scope(context.prior_outputs)
        if scope:
            results = self.vector_store.query(context.goal, top_k=self.top_k, repo=scope)
        else:
            results = self.vector_store.query(context.goal, top_k=self.top_k)
        # Keep the real code for later steps (GENERATE). Empty if nothing found.
        context.prior_outputs[REPO_CONTEXT_KEY] = format_chunks(results) if results else ""
        if not results:
            return "No matching chunks found for the goal."
        lines = [f"Retrieved {len(results)} chunks:"]
        for i, r in enumerate(results, start=1):
            meta = r.metadata or {}
            qname = meta.get("qualified_name", "<unknown>")
            fpath = meta.get("file_path", "<unknown>")
            lines.append(f"  {i}. {qname}  ({fpath})  distance={r.distance:.4f}")
        return "\n".join(lines)


class AnalyzeExecutor:
    """Runs the full RAG pipeline against the goal and returns the answer."""

    kind: StepKind = StepKind.ANALYZE

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def run(self, context: StepContext) -> str:
        scope = _repo_scope(context.prior_outputs)
        answer = (
            self.pipeline.ask(context.goal, repo=scope)
            if scope else self.pipeline.ask(context.goal)
        )
        return f"{answer.answer}\n\n(sources: {answer.source_count})"


class GitHubReadExecutor:
    """Reads a small snapshot from GitHub relevant to the goal."""

    kind: StepKind = StepKind.GITHUB_READ

    def __init__(self, github: GitHubAPI, coord: RepoCoord | None = None) -> None:
        self.github = github
        self.coord = coord

    def run(self, context: StepContext) -> str:
        # The repo named in the request wins; a repo set on the server is the fallback.
        target = context.prior_outputs.get(REPO_TARGET_KEY)
        coord = RepoCoord(owner=target.owner, repo=target.repo) if target is not None else self.coord
        if coord is None:
            raise StepSkipped(
                "GitHub read requested but no repository was configured. "
                "Send owner and repo with the request to enable it."
            )
        try:
            issues = self.github.list_issues(coord, state="open", per_page=5)
        except Exception as exc:
            raise RuntimeError(f"GitHub read failed: {exc}") from exc
        if not issues:
            return f"No open issues in {coord.slug()}."
        lines = [f"Open issues in {coord.slug()}:"]
        for issue in issues:
            lines.append(f"  #{issue.number}  {issue.title}  (by {issue.author})")
        return "\n".join(lines)


class GenerateExecutor:
    """Runs FixGenerator using the goal and real repository code.

    The code comes from the RETRIEVE step. If that step did not run, the
    executor searches the repository itself (same method as /fix/propose).
    """

    kind: StepKind = StepKind.GENERATE

    def __init__(
        self,
        generator: FixGenerator,
        vector_store: VectorStore | None = None,
        *,
        top_k: int = 5,
    ) -> None:
        self.generator = generator
        self.vector_store = vector_store
        self.top_k = top_k

    def run(self, context: StepContext) -> str:
        repo_context = self._get_repo_context(context)
        proposal = self.generator.propose(context.goal, repo_context)
        # Hand the proposal to later steps (validate, draft PR).
        context.prior_outputs[FIX_PROPOSAL_KEY] = proposal
        head = (
            f"Fix proposal (model={proposal.model}, "
            f"confidence={proposal.confidence:.2f}, "
            f"files_changed={proposal.files_changed}, "
            f"is_valid={proposal.is_valid})"
        )
        if repo_context.strip():
            context_line = "Context: repository code attached"
        else:
            context_line = "Context: none found, proposal is based on the goal only"
        body = proposal.explanation
        if not proposal.is_valid:
            body += f"\n\n[!] Validation error: {proposal.validation_error}"
        return (
            f"{head}\n{context_line}\nDiff SHA-256: {proposal.diff_sha256}\n\n"
            f"{body}\n\n{PATCH_MARKER}\n{proposal.diff}"
        )

    def _get_repo_context(self, context: StepContext) -> str:
        saved = context.prior_outputs.get(REPO_CONTEXT_KEY)
        if saved is not None:
            # The RETRIEVE step already searched (the text may be empty).
            return saved
        if self.vector_store is not None:
            return retrieve_repo_context(
                self.vector_store, context.goal, top_k=self.top_k,
                repo=_repo_scope(context.prior_outputs),
            )
        return ""


class ValidateExecutor:
    """Runs a ValidationPipeline against a FixProposal from prior_outputs."""

    kind: StepKind = StepKind.VALIDATE

    def __init__(self, pipeline: ValidationPipeline) -> None:
        self.pipeline = pipeline

    def run(self, context: StepContext) -> str:
        proposal = context.prior_outputs.get(FIX_PROPOSAL_KEY)
        if proposal is None:
            raise StepSkipped(
                "no fix proposal from an earlier step, so there is nothing to validate."
            )
        repo = context.prior_outputs.get(REPO_TARGET_KEY)
        if repo is not None:
            result: ValidationResult = self.pipeline.validate(proposal, repo=repo)
        else:
            result = self.pipeline.validate(proposal)
        # Hand the result to later steps (draft PR).
        context.prior_outputs[VALIDATION_RESULT_KEY] = result
        head = (
            f"Validation (passed={result.passed}, "
            f"score={result.score:.2f}, "
            f"checks={result.total_checks})"
        )
        lines = [head]
        if result.base_description:
            lines.append(f"Checked against: {result.base_description}")
        if result.diff_sha256:
            lines.append(f"Checked diff SHA-256: {result.diff_sha256}")
        if result.error:
            lines.append(f"[!] fatal: {result.error}")
        for c in result.checks:
            marker = "[skipped]" if c.skipped else ("[pass]" if c.passed else "[fail]")
            lines.append(f"  {marker} {c.name}: {c.message}")
        return "\n".join(lines)


class DraftPRExecutor:
    """Runs a DraftPRService against outputs of prior steps."""

    kind: StepKind = StepKind.DRAFT_PR

    def __init__(
        self,
        service: DraftPRService,
        *,
        owner: str,
        repo: str,
        base_branch: str = "main",
    ) -> None:
        if not owner or not repo:
            raise ValueError("owner and repo must be provided.")
        self.service = service
        self.owner = owner
        self.repo = repo
        self.base_branch = base_branch

    def run(self, context: StepContext) -> str:
        proposal = context.prior_outputs.get(FIX_PROPOSAL_KEY)
        validation = context.prior_outputs.get(VALIDATION_RESULT_KEY)
        if proposal is None:
            raise StepSkipped(
                "no fix proposal from an earlier step, so there is nothing to open a pull request for."
            )
        if validation is None:
            raise StepSkipped(
                "no validation result from an earlier step; validate the fix first."
            )
        request = DraftPRRequest(
            owner=self.owner,
            repo=self.repo,
            base_branch=self.base_branch,
            goal=context.goal,
            proposal_explanation=proposal.explanation,
            proposal_diff=proposal.diff,
            confidence=proposal.confidence,
            validation_passed=validation.passed,
            validation_score=validation.score,
        )
        result: DraftPRResult = self.service.create(request)
        lines = [f"Draft PR (created={result.created})"]
        if result.created:
            lines.append(f"  #{result.pr_number}  {result.title}")
            lines.append(f"  branch: {result.branch}")
            lines.append(f"  url: {result.pr_url}")
        else:
            if result.skipped_reason:
                lines.append(f"  skipped: {result.skipped_reason}")
            if result.error:
                lines.append(f"  error: {result.error}")
        return "\n".join(lines)


class PlannedExecutor:
    """An honest not-yet-implemented executor placeholder. Always reports SKIPPED."""

    def __init__(self, kind: StepKind, note: str) -> None:
        self.kind = kind
        self.note = note

    def run(self, context: StepContext) -> str:
        raise StepSkipped(f"[planned] {self.kind.value}: {self.note}")


def make_default_executors(
    vector_store: VectorStore,
    pipeline: RAGPipeline,
    github: GitHubAPI | None = None,
    coord: RepoCoord | None = None,
    generator: FixGenerator | None = None,
    validation_pipeline: ValidationPipeline | None = None,
    draft_pr_service: DraftPRService | None = None,
    draft_pr_owner: str | None = None,
    draft_pr_repo: str | None = None,
    draft_pr_base_branch: str = "main",
) -> dict:
    """Build the default map of StepKind -> StepExecutor."""
    executors: dict = {
        StepKind.RETRIEVE: RetrieveExecutor(vector_store),
        StepKind.ANALYZE: AnalyzeExecutor(pipeline),
    }
    if github is not None:
        executors[StepKind.GITHUB_READ] = GitHubReadExecutor(github, coord=coord)
    else:
        executors[StepKind.GITHUB_READ] = PlannedExecutor(
            StepKind.GITHUB_READ,
            "no GitHub client configured on this server.",
        )
    if generator is not None:
        executors[StepKind.GENERATE] = GenerateExecutor(generator, vector_store=vector_store)
    else:
        executors[StepKind.GENERATE] = PlannedExecutor(
            StepKind.GENERATE, "wire a FixGenerator to enable diff generation.",
        )
    if validation_pipeline is not None:
        executors[StepKind.VALIDATE] = ValidateExecutor(validation_pipeline)
    else:
        executors[StepKind.VALIDATE] = PlannedExecutor(
            StepKind.VALIDATE, "wire a ValidationPipeline to enable fix validation.",
        )
    executors[StepKind.HUMAN_APPROVAL] = PlannedExecutor(
        StepKind.HUMAN_APPROVAL, "human approval is handled outside the engine.",
    )
    if draft_pr_service is not None and draft_pr_owner and draft_pr_repo:
        executors[StepKind.DRAFT_PR] = DraftPRExecutor(
            draft_pr_service,
            owner=draft_pr_owner,
            repo=draft_pr_repo,
            base_branch=draft_pr_base_branch,
        )
    else:
        executors[StepKind.DRAFT_PR] = PlannedExecutor(
            StepKind.DRAFT_PR,
            "wire a DraftPRService + owner/repo to enable draft PR creation.",
        )
    return executors
