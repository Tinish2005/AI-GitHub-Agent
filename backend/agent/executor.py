"""Step executors for the ExecutionEngine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from backend.agent.models import PlanStep, StepKind
from backend.github.client import GitHubAPI
from backend.github.models import RepoCoord
from backend.indexing.vector_store import VectorStore
from backend.rag.pipeline import RAGPipeline


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


def make_default_executors(
    vector_store: VectorStore,
    pipeline: RAGPipeline,
    github: GitHubAPI | None = None,
    coord: RepoCoord | None = None,
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
    executors[StepKind.GENERATE] = PlannedExecutor(
        StepKind.GENERATE, "Loop 11 will implement LLM-based fix generation.",
    )
    executors[StepKind.VALIDATE] = PlannedExecutor(
        StepKind.VALIDATE, "Loop 12 will implement build/tests/lint validation.",
    )
    executors[StepKind.HUMAN_APPROVAL] = PlannedExecutor(
        StepKind.HUMAN_APPROVAL, "human approval is handled outside the engine.",
    )
    executors[StepKind.DRAFT_PR] = PlannedExecutor(
        StepKind.DRAFT_PR, "Loop 13 will implement draft PR creation via GitHub API.",
    )
    return executors
