"""Shared helper: turn vector-store search results into code context text.

Used by the /fix/* routes (via main.py) and by the Execute steps, so both
give the AI the same real source code for the same goal.
"""

from __future__ import annotations

from backend.indexing.vector_store import EmbeddingMismatchError


def format_chunks(results) -> str:
    """Format search results as numbered blocks that include the source code."""
    blocks = []
    for i, r in enumerate(results, start=1):
        meta = r.metadata or {}
        qname = meta.get("qualified_name", "<unknown>")
        fpath = meta.get("file_path", "<unknown>")
        blocks.append(f"[{i}] {qname} ({fpath})\n{r.document}")
    return "\n\n".join(blocks)


def retrieve_repo_context(vector_store, goal, top_k=5, repo=None) -> str:
    """Search the indexed repo for code relevant to the goal, return as text.

    With `repo` ("owner/name") only that repo's chunks are searched.
    Returns an empty string when nothing is found or the search fails.
    A model mismatch is never hidden: it is raised.
    """
    try:
        if repo:
            results = vector_store.query(goal, top_k=top_k, repo=repo)
        else:
            results = vector_store.query(goal, top_k=top_k)
    except EmbeddingMismatchError:
        raise
    except Exception:
        return ""
    if not results:
        return ""
    return format_chunks(results)
