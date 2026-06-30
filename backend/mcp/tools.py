"""
Tool registry + concrete tool implementations.

Each tool has:
    - A stable name (client-facing).
    - A JSON-Schema-ish input_schema so clients know what to send.
    - An execute function that takes a dict of params and returns
      a plain string (we wrap it into MCP content blocks at the
      server layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from backend.indexing.vector_store import VectorStore
from backend.rag.pipeline import RAGPipeline
from backend.github.client import GitHubAPI
from backend.github.models import RepoCoord



ToolExecutor = Callable[[dict], str]


@dataclass(frozen=True)
class Tool:
    """A registered tool."""

    name: str
    description: str
    input_schema: dict
    execute: ToolExecutor


class ToolRegistry:
    """Holds tools and looks them up by name."""

    def __init__(self) -> None:
        self._tools: dict = {}

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry."""
        if not tool.name:
            raise ValueError("Tool name must not be empty.")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Return a tool by name, or None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """Return tools in stable alphabetical order."""
        return sorted(self._tools.values(), key=lambda t: t.name)


def make_search_code_tool(store: VectorStore) -> Tool:
    """Tool: search the codebase by natural-language query."""

    def execute(params: dict) -> str:
        query = str(params.get("query", "")).strip()
        if not query:
            raise ValueError("Param 'query' is required and must be non-empty.")
        top_k = int(params.get("top_k", 5))
        if top_k < 1 or top_k > 20:
            raise ValueError("Param 'top_k' must be between 1 and 20.")

        
        
        results = store.query(query, top_k=top_k)
        if not results:
            return "No matching chunks found."

        lines = [f"Found {len(results)} matches:"]
        for i, r in enumerate(results, start=1):
            meta = r.metadata or {}
            qname = meta.get("qualified_name", "<unknown>")
            fpath = meta.get("file_path", "<unknown>")
            start = meta.get("start_line", "?")
            end = meta.get("end_line", "?")
            lines.append(f"{i}. {qname}  ({fpath}:{start}-{end})  distance={r.distance:.4f}")
        return "\n".join(lines)

    return Tool(
        name="search_code",
        description="Search the indexed codebase for chunks matching a natural-language query.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["query"],
        },
        execute=execute,
    )

    
def make_get_chunk_tool(store: VectorStore) -> Tool:
    """Tool: fetch a single chunk by its content-hash id."""

    def execute(params: dict) -> str:
        chunk_id = str(params.get("chunk_id", "")).strip()
        if not chunk_id:
            raise ValueError("Param 'chunk_id' is required and must be non-empty.")

        store._ensure_loaded()
        try:
            idx = store._ids.index(chunk_id)
        except ValueError as exc:
            raise ValueError(f"chunk_id '{chunk_id}' not found.") from exc

        document = store._documents[idx]
        meta = store._metadatas[idx]
        header = (
            f"{meta.get('qualified_name', '<unknown>')}  "
            f"({meta.get('file_path', '<unknown>')}:"
            f"{meta.get('start_line', '?')}-{meta.get('end_line', '?')})"
        )
        return f"{header}\n\n{document}"

    return Tool(
        name="get_chunk",
        description="Fetch a single chunk by its content-hash id.",
        input_schema={
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string", "description": "SHA-256 content hash."},
            },
            "required": ["chunk_id"],
        },
        execute=execute,
    )



def make_ask_codebase_tool(pipeline: RAGPipeline) -> Tool:
    """Tool: full RAG Q&A on the indexed codebase."""

    def execute(params: dict) -> str:
        question = str(params.get("question", "")).strip()
        if not question:
            raise ValueError("Param 'question' is required and must be non-empty.")
        top_k = int(params.get("top_k", pipeline.top_k))
        if top_k < 1 or top_k > 20:
            raise ValueError("Param 'top_k' must be between 1 and 20.")

        original_top_k = pipeline.top_k
        try:
            pipeline.top_k = top_k
            answer = pipeline.ask(question)
        finally:
            pipeline.top_k = original_top_k

        citation_lines = [
            f"[{i + 1}] {s.qualified_name} ({s.file_path}:{s.start_line}-{s.end_line})"
            for i, s in enumerate(answer.sources)
        ]
        citations = "\n".join(citation_lines) if citation_lines else "(no sources)"
        return f"{answer.answer}\n\nSources:\n{citations}"

    return Tool(
        name="ask_codebase",
        description="Ask a natural-language question about the indexed codebase.",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The natural-language question."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["question"],
        },
        execute=execute,
    )

    

def build_default_registry(
    store: VectorStore,
    pipeline: RAGPipeline,
    github: GitHubAPI | None = None,
) -> ToolRegistry:
    """Build a registry pre-populated with the standard agent toolset."""
    reg = ToolRegistry()
    reg.register(make_search_code_tool(store))
    reg.register(make_get_chunk_tool(store))
    reg.register(make_ask_codebase_tool(pipeline))
    if github is not None:
        reg.register(make_github_get_file_tool(github))
        reg.register(make_github_list_issues_tool(github))
        reg.register(make_github_get_pr_tool(github))
    return reg

    

def make_github_get_file_tool(github: GitHubAPI) -> Tool:
    """Tool: fetch a single file from a GitHub repo."""

    def execute(params: dict) -> str:
        owner = str(params.get("owner", "")).strip()
        repo = str(params.get("repo", "")).strip()
        path = str(params.get("path", "")).strip()
        ref = params.get("ref")
        if not owner or not repo or not path:
            raise ValueError("Params 'owner', 'repo', and 'path' are required.")
        if ref is not None:
            ref = str(ref).strip() or None
        try:
            f = github.get_file(RepoCoord(owner=owner, repo=repo), path, ref=ref)
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc
        header = f"{owner}/{repo}:{f.path}  (sha={f.sha[:8]}, size={f.size})"
        return f"{header}\n\n{f.content}"

    return Tool(
        name="github_get_file",
        description="Fetch a single file from a GitHub repository.",
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repo owner or org login."},
                "repo": {"type": "string", "description": "Repository name."},
                "path": {"type": "string", "description": "File path within the repo."},
                "ref": {"type": "string", "description": "Optional branch / tag / SHA."},
            },
            "required": ["owner", "repo", "path"],
        },
        execute=execute,
    )


def make_github_list_issues_tool(github: GitHubAPI) -> Tool:
    """Tool: list issues on a GitHub repo."""

    def execute(params: dict) -> str:
        owner = str(params.get("owner", "")).strip()
        repo = str(params.get("repo", "")).strip()
        state = str(params.get("state", "open")).strip() or "open"
        per_page = int(params.get("per_page", 30))
        if not owner or not repo:
            raise ValueError("Params 'owner' and 'repo' are required.")
        if state not in ("open", "closed", "all"):
            raise ValueError("state must be one of: open, closed, all")
        if per_page < 1 or per_page > 100:
            raise ValueError("per_page must be between 1 and 100")
        issues = github.list_issues(
            RepoCoord(owner=owner, repo=repo),
            state=state,
            per_page=per_page,
        )
        if not issues:
            return f"No {state} issues found in {owner}/{repo}."
        lines = [f"Found {len(issues)} {state} issue(s) in {owner}/{repo}:"]
        for i in issues:
            lines.append(f"  #{i.number}  [{i.state}]  {i.title}  (by {i.author})")
        return "\n".join(lines)

    return Tool(
        name="github_list_issues",
        description="List issues on a GitHub repository.",
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"},
                "per_page": {"type": "integer", "minimum": 1, "maximum": 100, "default": 30},
            },
            "required": ["owner", "repo"],
        },
        execute=execute,
    )


def make_github_get_pr_tool(github: GitHubAPI) -> Tool:
    """Tool: fetch a pull request by number."""

    def execute(params: dict) -> str:
        owner = str(params.get("owner", "")).strip()
        repo = str(params.get("repo", "")).strip()
        number = int(params.get("number", 0))
        if not owner or not repo:
            raise ValueError("Params 'owner' and 'repo' are required.")
        if number < 1:
            raise ValueError("PR 'number' must be >= 1.")
        try:
            pr = github.get_pr(RepoCoord(owner=owner, repo=repo), number)
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc
        merged_str = "yes" if pr.merged else "no"
        return (
            f"PR #{pr.number}  [{pr.state}]  {pr.title}\n"
            f"Author: {pr.author}\n"
            f"Branch: {pr.head} -> {pr.base}\n"
            f"Merged: {merged_str}\n"
            f"URL: {pr.url}\n\n"
            f"{pr.body}"
        )

    return Tool(
        name="github_get_pr",
        description="Fetch a pull request by its number.",
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "number": {"type": "integer", "minimum": 1},
            },
            "required": ["owner", "repo", "number"],
        },
        execute=execute,
    )


