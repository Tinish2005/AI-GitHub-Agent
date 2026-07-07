"""Step executors for the ExecutionEngine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from backend.agent.models import PlanStep, StepKind
from backend.github.client import GitHubAPI
from backend.github.models import RepoCoord
from backend.indexing.vector_store import VectorStore
from backend.rag.pipeline import RAGPipeline
from backend.agent.fix_generator import FixGenerator
from backend.agent.validation_models import ValidationResult
from backend.agent.validation_pipeline import ValidationPipeline



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
        results = self.vector_store.query(context.goal, top_k=self.top_k)
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
        answer = self.pipeline.ask(context.goal)
        return f"{answer.answer}\n\n(sources: {answer.source_count})"


class GitHubReadExecutor:
    """Reads a small snapshot from GitHub relevant to the goal."""

    kind: StepKind = StepKind.GITHUB_READ

    def __init__(self, github: GitHubAPI, coord: RepoCoord | None = None) -> None:
        self.github = github
        self.coord = coord

    def run(self, context: StepContext) -> str:
        if self.coord is None:
            return (
                "GitHub read requested but no repository was configured. "
                "Wire a RepoCoord into the engine to enable live reads."
            )
        try:
            issues = self.github.list_issues(self.coord, state="open", per_page=5)
        except Exception as exc:
            raise RuntimeError(f"GitHub read failed: {exc}") from exc
        if not issues:
            return f"No open issues in {self.coord.slug()}."
        lines = [f"Open issues in {self.coord.slug()}:"]
        for issue in issues:
            lines.append(f"  #{issue.number}  {issue.title}  (by {issue.author})")
        return "\n".join(lines)


class PlannedExecutor:
    """A honest not-yet-implemented executor for Loops 11-13 kinds."""

    def __init__(self, kind: StepKind, note: str) -> None:
        self.kind = kind
        self.note = note

    def run(self, context: StepContext) -> str:
        return f"[planned] {self.kind.value}: {self.note}"

class GenerateExecutor:
    """
    Runs FixGenerator against the goal and prior retrieval output.

    Uses the output of the previous RETRIEVE step (step id 1 by convention)
    as the code context. If no prior retrieval output exists, passes an
    empty context and lets the LLM decide what to do.
    """

    kind: StepKind = StepKind.GENERATE

    def __init__(self, generator: FixGenerator) -> None:
        self.generator = generator

    def run(self, context: StepContext) -> str:
        prior_context = self._pick_context(context.prior_outputs)
        proposal = self.generator.propose(context.goal, prior_context)
        head = (
            f"Fix proposal (model={proposal.model}, "
            f"confidence={proposal.confidence:.2f}, "
            f"files_changed={proposal.files_changed}, "
            f"is_valid={proposal.is_valid})"
        )
        body = proposal.explanation
        if not proposal.is_valid:
            body += f"\n\n[!] Validation error: {proposal.validation_error}"
        return f"{head}\n\n{body}"

    @staticmethod
    def _pick_context(prior_outputs: dict) -> str:
        if not prior_outputs:
            return ""
        # Prefer the RETRIEVE step (id 1 by convention). Fall back to newest.
        if 1 in prior_outputs:
            return str(prior_outputs[1])
        latest_id = max(prior_outputs.keys())
        return str(prior_outputs[latest_id])


class ValidateExecutor:
    """
    Runs a ValidationPipeline against a FixProposal produced by an
    earlier GENERATE step.

    Convention: the GENERATE step stores its FixProposal into
    prior_outputs[<step.id>] as its string summary. For validation we
    need the actual FixProposal object, so this executor takes the
    generator's proposal directly via a shared reference in
    context.prior_outputs under the key 'fix_proposal' when available.
    Otherwise it reports "no proposal to validate".
    """

    kind: StepKind = StepKind.VALIDATE

    def __init__(self, pipeline: ValidationPipeline) -> None:
        self.pipeline = pipeline

    def run(self, context: StepContext) -> str:
        proposal = context.prior_outputs.get("fix_proposal")
        if proposal is None:
            return "[validate] no FixProposal found in prior outputs; skipping."
        result: ValidationResult = self.pipeline.validate(proposal)
        head = (
            f"Validation (passed={result.passed}, "
            f"score={result.score:.2f}, "
            f"checks={result.total_checks})"
        )
        lines = [head]
        if result.error:
            lines.append(f"[!] fatal: {result.error}")
        for c in result.checks:
            marker = "[skipped]" if c.skipped else ("[pass]" if c.passed else "[fail]")
            lines.append(f"  {marker} {c.name}: {c.message}")
        return "\n".join(lines)


def make_default_executors(
    vector_store: VectorStore,
    pipeline: RAGPipeline,
    github: GitHubAPI | None = None,
    coord: RepoCoord | None = None,
    generator: FixGenerator | None = None,
    validation_pipeline: ValidationPipeline | None = None,
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
        executors[StepKind.GENERATE] = GenerateExecutor(generator)
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
    executors[StepKind.DRAFT_PR] = PlannedExecutor(
        StepKind.DRAFT_PR, "Loop 13 will implement draft PR creation via GitHub API.",
    )
    return executors