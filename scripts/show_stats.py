"""Print what is indexed in your local store (no server, no Gemini, no GitHub).

Run from the project root, in PowerShell:

    python scripts\\show_stats.py
    python scripts\\show_stats.py --repo psf/requests --top 10
    python scripts\\show_stats.py --json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def render(stats) -> str:
    """Format a StatsResponse as plain text."""
    lines = [
        f"Repos: {stats.total_repos}   Files: {stats.total_files}   Chunks: {stats.total_chunks}"
    ]
    if stats.embedding_dimension:
        lines.append(
            f"Embeddings: {stats.embedding_dimension} numbers per chunk, "
            f"model {stats.embedding_model or 'not recorded (index built before model tracking)'}"
        )
    if not stats.repos:
        lines.append("Nothing is indexed (or no repo matched).")
    for r in stats.repos:
        lines.append("")
        lines.append(f"== {r.repo}" + ("  [DEMO]" if r.is_demo else ""))
        lines.append(f"   branch: {r.ref or 'unknown'}   last synced: {r.last_synced or 'unknown'}")
        lines.append(f"   {r.files} files, {r.chunks} chunks, types: {r.chunks_by_type}")
        lines.append("   chunks per file:")
        for f in r.chunks_per_file:
            lines.append(f"     {f.chunks:4}  {f.file_path}")
        lines.append(f"   imports between files: {r.import_edge_count}")
        for e in r.import_edges[:10]:
            lines.append(f"     {e.source} -> {e.target}")
        if r.top_external_imports:
            names = ", ".join(f"{i.name}({i.files})" for i in r.top_external_imports)
            lines.append(f"   most used outside imports: {names}")
        if not r.tracked:
            lines.append("   (saved before repo labels existed: index the repo again to label it)")
    return "\n".join(lines)


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="Show local index statistics.")
    parser.add_argument("--repo", default=None, help="only this repo, as owner/name")
    parser.add_argument("--top", type=int, default=20, help="rows per list")
    parser.add_argument("--json", action="store_true", help="print the raw JSON")
    args = parser.parse_args(argv[1:])

    from backend.config import get_settings
    from backend.indexing.stats import compute_stats
    from backend.indexing.vector_store import VectorStore

    store = VectorStore(persist_directory=get_settings().vector_db_path)
    stats = compute_stats(store.all_metadata(), repo=args.repo, top=args.top)
    info = store.embedding_info()
    stats.embedding_model = info["model"]
    stats.embedding_dimension = info["dimension"]
    print(stats.model_dump_json(indent=2) if args.json else render(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
