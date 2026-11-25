"""Build the demo seed file (run once on your computer, then commit the file).

Run from the project root, in PowerShell:

    python scripts\\build_demo_seed.py

It indexes the public repo psf/requests at ONE fixed commit, keeps only the
files under src/, marks every chunk as demo, and saves a compressed seed file
(default: demo_seed\\psf_requests.json.gz) plus a small manifest next to it.

It needs your Gemini key (from your .env or the GEMINI_API_KEY variable) and
takes about 5 minutes. It uses a temporary store, so your live store is not
touched. Delete your old local vector_db first if you want a clean local copy.

Options: --owner, --repo, --include (path prefix, default src/), --max-files, --out
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build(owner: str, repo: str, include: str, out_path: Path, *, store, indexer, provider) -> dict:
    """Index the repo at one commit into `store`, then write the seed file."""
    from backend.github.models import RepoCoord
    from backend.indexing.demo_seed import DemoSeedError, export_seed

    snapshot = provider.fetch(owner, repo, None, [])
    commit = snapshot.commit
    branch = indexer._default_branch(owner, repo)
    result = indexer.index(
        RepoCoord(owner=owner, repo=repo),
        ref=commit,
        include_prefixes=(include,) if include else None,
        tags={"ref": branch, "commit": commit, "demo": "true"},
    )
    if result.chunks_indexed == 0:
        raise DemoSeedError(
            f"No chunks were indexed. Does {owner}/{repo} have Python files under '{include}'?"
        )
    return export_seed(
        store, out_path,
        {"include": include, "branch": branch, "files_indexed": result.files_indexed},
    )


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="Build the demo seed file.")
    parser.add_argument("--owner", default="psf")
    parser.add_argument("--repo", default="requests")
    parser.add_argument("--include", default="src/", help="only index files under this path")
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--out", default=None, help="seed file to write")
    args = parser.parse_args(argv[1:])

    from backend.config import get_settings
    from backend.github.repo_indexer import GitHubRepoIndexer
    from backend.github.repo_snapshot import GitHubSnapshotProvider, SnapshotError
    from backend.indexing.demo_seed import DemoSeedError
    from backend.indexing.embedding_factory import build_embedding_service
    from backend.indexing.vector_store import EmbeddingMismatchError, VectorStore

    settings = get_settings()
    if not settings.gemini_api_key.get_secret_value():
        print("No Gemini key found. Put GEMINI_API_KEY in your .env file, or set it in this window.")
        return 2
    out_path = Path(args.out) if args.out else settings.demo_seed_path
    token = settings.github_token.get_secret_value() or None

    print(f"Building the demo seed for {args.owner}/{args.repo} (files under '{args.include}').")
    print("This takes about 5 minutes and uses Gemini quota. Your live store is not touched.")
    with tempfile.TemporaryDirectory(prefix="demo_seed_build_") as tmp:
        store = VectorStore(
            persist_directory=Path(tmp),
            embedding_service=build_embedding_service(settings),
        )
        indexer = GitHubRepoIndexer(token=token, vector_store=store, max_files=args.max_files)
        try:
            manifest = build(
                args.owner, args.repo, args.include, out_path,
                store=store, indexer=indexer, provider=GitHubSnapshotProvider(token=token),
            )
        except (DemoSeedError, SnapshotError, EmbeddingMismatchError) as exc:
            print(f"Build failed: {exc}")
            return 1

    size_mb = out_path.stat().st_size / 1e6
    print("\nDone.")
    print(f"  file:      {out_path}  ({size_mb:.1f} MB)")
    print(f"  repo:      {manifest['repo']}   commit: {manifest['commit'][:7]}")
    print(f"  chunks:    {manifest['chunks']} from {manifest['files']} files")
    print(f"  embedding: {manifest['dimension']} numbers per chunk, model {manifest['model']}")
    print("\nNext steps:")
    print("  1. python scripts\\check_demo_seed.py        (checks the file and your .gitignore)")
    print("  2. Commit the demo_seed folder with your code and make sure it is on GitHub.")
    print(f"  3. On Render and locally, set GEMINI_EMBEDDING_MODEL={manifest['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
