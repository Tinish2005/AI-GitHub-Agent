"""
Context assembler.

Takes raw retrieval results from the vector store and prepares them
for the LLM:
    - Dedups by chunk_id.
    - Ranks by distance (lower first).
    - Truncates to a character budget so we don't blow the LLM's
      context window.
    - Renders a numbered, citation-friendly context block.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.indexing.vector_store import QueryResult


DEFAULT_CONTEXT_BUDGET_CHARS: int = 6000


@dataclass(frozen=True)
class AssembledContext:
    """The fully-prepared context block plus the chunks that were kept."""

    text: str
    kept: tuple[QueryResult, ...]
    used_chars: int


def assemble_context(
    results: list[QueryResult],
    *,
    budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
) -> AssembledContext:
    """
    Build a single string of code chunks for the LLM, within the budget.

    Chunks are kept in distance order (best first) and added until the
    next one would exceed `budget_chars`. Each rendered chunk is
    numbered so the LLM (and the user) can cite it as `[1]`, `[2]`, ...
    """
    if budget_chars < 1:
        raise ValueError("budget_chars must be >= 1.")

    deduped: list[QueryResult] = []
    seen: set[str] = set()
    for r in sorted(results, key=lambda x: x.distance):
        if r.chunk_id in seen:
            continue
        seen.add(r.chunk_id)
        deduped.append(r)

    blocks: list[str] = []
    kept: list[QueryResult] = []
    used = 0
    for index, r in enumerate(deduped, start=1):
        block = _render_block(index, r)
        if used + len(block) > budget_chars and kept:
            break
        blocks.append(block)
        kept.append(r)
        used += len(block)

    return AssembledContext(
        text="\n\n".join(blocks),
        kept=tuple(kept),
        used_chars=used,
    )


def _render_block(index: int, r: QueryResult) -> str:
    """Render one retrieved chunk as a citation-numbered code block."""
    meta = r.metadata or {}
    qname = meta.get("qualified_name", "<unknown>")
    fpath = meta.get("file_path", "<unknown>")
    start = meta.get("start_line", "?")
    end = meta.get("end_line", "?")
    return (
        f"[{index}] {qname}  ({fpath}:{start}-{end})\n"
        f"```\n{r.document.rstrip()}\n```"
    )