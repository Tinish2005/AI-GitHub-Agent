"""Statistics about what is indexed, computed from the saved chunk metadata.

Everything here is read-only. It works on the list of metadata dicts that the
vector store keeps (one per chunk) and never calls Gemini or GitHub.

Chunks are grouped by the `repo` label written at index time ("owner/name").
Chunks saved before labels existed are grouped under UNTRACKED_LABEL.

Import links are best effort and Python only. An imported name counts as a
link when it matches a file of the same repo (next to the importing file
first, then by dotted module path). Names that match no file are counted as
outside imports; that list includes the standard library.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from backend.indexing.demo_seed import DemoSeedStatus


UNTRACKED_LABEL = "(indexed before tracking)"
EDGE_LIMIT = 500

_AMBIGUOUS = object()


class FileStat(BaseModel):
    file_path: str
    chunks: int = Field(ge=0)


class ImportEdge(BaseModel):
    source: str = Field(description="File that contains the import.")
    target: str = Field(description="File of the same repo that it imports.")


class ImportStat(BaseModel):
    name: str
    files: int = Field(ge=0, description="How many files import this name.")


class RepoStats(BaseModel):
    repo: str = Field(description='"owner/name", or "(indexed before tracking)".')
    tracked: bool = Field(description="False for chunks saved before repo labels existed.")
    ref: str = Field(default="", description="Branch that was indexed, if known.")
    commit: str = Field(default="", description="Commit that was indexed, if known (demo repo).")
    is_demo: bool = Field(default=False)
    last_synced: str | None = Field(default=None, description="Latest index time (UTC, ISO 8601).")
    files: int = Field(ge=0)
    chunks: int = Field(ge=0)
    chunks_by_type: dict[str, int] = Field(default_factory=dict)
    chunks_per_file: list[FileStat] = Field(default_factory=list)
    import_edge_count: int = Field(ge=0, default=0)
    import_edges: list[ImportEdge] = Field(default_factory=list)
    top_external_imports: list[ImportStat] = Field(default_factory=list)


class StatsResponse(BaseModel):
    total_repos: int = Field(ge=0, description="Tracked repos only.")
    total_files: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    repos: list[RepoStats] = Field(default_factory=list)
    embedding_model: str | None = Field(default=None, description="Model the index was built with, if known.")
    embedding_dimension: int | None = Field(default=None, description="Numbers per chunk vector, if any chunks.")
    limits: dict | None = Field(
        default=None, description="Public indexing and AI limits, and what is left today."
    )
    demo_seed: DemoSeedStatus | None = Field(
        default=None, description="What happened to the demo seed at startup (None if startup did not run)."
    )


def compute_stats(metadatas: list, *, repo: str | None = None, top: int = 20) -> StatsResponse:
    """Build the statistics for every repo in the store (or just `repo`)."""
    groups: dict = defaultdict(list)
    for meta in metadatas or []:
        if not isinstance(meta, dict):
            continue
        label = str(meta.get("repo") or "").strip() or UNTRACKED_LABEL
        groups[label].append(meta)

    if repo and repo.strip():
        wanted = repo.strip().lower()
        groups = {k: v for k, v in groups.items() if k.lower() == wanted}

    ordered = sorted(groups.items(), key=lambda kv: (kv[0] == UNTRACKED_LABEL, kv[0].lower()))
    repos = [_group_stats(label, items, top) for label, items in ordered]
    return StatsResponse(
        total_repos=sum(1 for r in repos if r.tracked),
        total_files=sum(r.files for r in repos),
        total_chunks=sum(r.chunks for r in repos),
        repos=repos,
    )


def _norm(path: object) -> str:
    return str(path).replace("\\", "/")


def _group_stats(label: str, items: list, top: int) -> RepoStats:
    per_file: Counter = Counter()
    types: Counter = Counter()
    refs: Counter = Counter()
    commits: Counter = Counter()
    synced: list = []
    imports_by_file: dict = defaultdict(set)
    demo = False

    for meta in items:
        path = _norm(meta.get("file_path") or "<unknown>")
        root = _norm(meta.get("root") or "").rstrip("/")
        if root and path.startswith(root + "/"):
            path = path[len(root) + 1:]
        per_file[path] += 1
        types[str(meta.get("chunk_type") or "unknown")] += 1
        if meta.get("ref"):
            refs[str(meta["ref"])] += 1
        if meta.get("commit"):
            commits[str(meta["commit"])] += 1
        if meta.get("indexed_at"):
            synced.append(str(meta["indexed_at"]))
        if str(meta.get("demo") or "").lower() == "true":
            demo = True
        imports_by_file[path].update(
            name for name in str(meta.get("imports") or "").split(",") if name
        )

    paths = set(per_file)
    index = _module_index(paths)
    edges: set = set()
    external: Counter = Counter()
    for path, names in imports_by_file.items():
        outside: set = set()
        for name in names:
            target = _resolve(name, path, paths, index)
            if target is _AMBIGUOUS:
                continue
            if target is None:
                outside.add(name.split(".")[0])
            elif target != path:
                edges.add((path, target))
        external.update(outside)

    sorted_edges = sorted(edges)
    file_rows = sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    external_rows = sorted(external.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return RepoStats(
        repo=label,
        tracked=label != UNTRACKED_LABEL,
        ref=refs.most_common(1)[0][0] if refs else "",
        commit=commits.most_common(1)[0][0] if commits else "",
        is_demo=demo,
        last_synced=max(synced) if synced else None,
        files=len(per_file),
        chunks=len(items),
        chunks_by_type=dict(sorted(types.items())),
        chunks_per_file=[FileStat(file_path=p, chunks=n) for p, n in file_rows],
        import_edge_count=len(sorted_edges),
        import_edges=[ImportEdge(source=s, target=t) for s, t in sorted_edges[:EDGE_LIMIT]],
        top_external_imports=[ImportStat(name=n, files=c) for n, c in external_rows],
    )


def _module_index(paths: set) -> dict:
    """Map every dotted suffix of a module path to the files that have it."""
    index: dict = defaultdict(list)
    for path in sorted(paths):
        if not path.endswith(".py"):
            continue
        parts = path[:-3].split("/")
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        for i in range(len(parts)):
            index[".".join(parts[i:])].append(path)
    return index


def _resolve(name: str, path: str, paths: set, index: dict):
    """Return the file an import refers to, None (not in this repo) or _AMBIGUOUS."""
    rel = name.replace(".", "/")
    folder = path.rsplit("/", 1)[0] if "/" in path else ""
    prefix = f"{folder}/" if folder else ""
    for candidate in (f"{prefix}{rel}.py", f"{prefix}{rel}/__init__.py"):
        if candidate in paths:
            return candidate
    hits = index.get(name, [])
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return _AMBIGUOUS
    return None
