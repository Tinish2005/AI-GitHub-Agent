"""Loop 6 tests: index statistics (OFFLINE, no Gemini, no GitHub)."""

from __future__ import annotations

import pytest

from backend.indexing.stats import UNTRACKED_LABEL, compute_stats


def _m(file_path, *, repo="o/r", ref="main", at="2026-10-01T10:00:00+00:00",
       type_="function", imports="", demo="", root="", name="f"):
    meta = {
        "qualified_name": name, "file_path": file_path, "chunk_type": type_,
        "imports": imports,
    }
    if repo is not None:
        meta["repo"] = repo
    if ref is not None:
        meta["ref"] = ref
    if at is not None:
        meta["indexed_at"] = at
    if demo:
        meta["demo"] = demo
    if root:
        meta["root"] = root
    return meta


SAMPLE = [
    _m("src/pkg/a.py", imports="os,pkg.b,c", type_="function"),
    _m("src/pkg/a.py", imports="os,pkg.b,c", type_="method"),
    _m("src/pkg/b.py", imports="json,requests.auth"),
    _m("src/pkg/c.py", imports="os"),
]


def test_empty_store_gives_an_empty_documented_result() -> None:
    stats = compute_stats([])
    assert stats.model_dump() == {
        "total_repos": 0, "total_files": 0, "total_chunks": 0, "repos": [],
        "embedding_model": None, "embedding_dimension": None, "demo_seed": None, "limits": None,
    }


def test_none_and_junk_entries_are_ignored() -> None:
    stats = compute_stats(None)  # type: ignore[arg-type]
    assert stats.total_chunks == 0
    stats = compute_stats([None, "x", 3, _m("a.py")])  # type: ignore[list-item]
    assert stats.total_chunks == 1


def test_counts_files_chunks_and_types() -> None:
    stats = compute_stats(SAMPLE)
    assert (stats.total_repos, stats.total_files, stats.total_chunks) == (1, 3, 4)
    repo = stats.repos[0]
    assert (repo.repo, repo.tracked, repo.files, repo.chunks) == ("o/r", True, 3, 4)
    assert repo.chunks_by_type == {"function": 3, "method": 1}
    assert [(f.file_path, f.chunks) for f in repo.chunks_per_file] == [
        ("src/pkg/a.py", 2), ("src/pkg/b.py", 1), ("src/pkg/c.py", 1),
    ]


def test_chunks_are_grouped_by_repo_and_old_chunks_are_untracked() -> None:
    data = [_m("a.py", repo="x/one"), _m("b.py", repo="y/two"), _m("c.py", repo=None, ref=None, at=None)]
    stats = compute_stats(data)
    assert [r.repo for r in stats.repos] == ["x/one", "y/two", UNTRACKED_LABEL]
    assert stats.total_repos == 2
    untracked = stats.repos[-1]
    assert untracked.tracked is False
    assert untracked.last_synced is None
    assert untracked.ref == ""


def test_blank_repo_label_counts_as_untracked() -> None:
    assert compute_stats([_m("a.py", repo="   ")]).repos[0].repo == UNTRACKED_LABEL


def test_last_synced_is_the_latest_time_and_ref_the_most_common() -> None:
    data = [
        _m("a.py", at="2026-10-01T10:00:00+00:00", ref="main"),
        _m("b.py", at="2026-10-05T09:30:00+00:00", ref="main"),
        _m("c.py", at="2026-09-20T09:30:00+00:00", ref="dev"),
    ]
    repo = compute_stats(data).repos[0]
    assert repo.last_synced == "2026-10-05T09:30:00+00:00"
    assert repo.ref == "main"


def test_demo_flag_is_false_by_default_and_true_when_marked() -> None:
    assert compute_stats([_m("a.py")]).repos[0].is_demo is False
    assert compute_stats([_m("a.py"), _m("b.py", demo="true")]).repos[0].is_demo is True
    assert compute_stats([_m("a.py", demo="false")]).repos[0].is_demo is False


def test_windows_paths_are_shown_with_forward_slashes() -> None:
    repo = compute_stats([_m("src\\pkg\\a.py")]).repos[0]
    assert repo.chunks_per_file[0].file_path == "src/pkg/a.py"


def test_clone_root_is_stripped_from_paths() -> None:
    root = "C:\\cache\\psf_requests"
    repo = compute_stats([_m(root + "\\src\\a.py", root=root)]).repos[0]
    assert repo.chunks_per_file[0].file_path == "src/a.py"


def test_top_limits_the_lists_but_not_the_totals() -> None:
    data = [_m(f"f{i}.py") for i in range(10)]
    repo = compute_stats(data, top=3).repos[0]
    assert len(repo.chunks_per_file) == 3
    assert repo.files == 10 and repo.chunks == 10


def test_repo_filter_is_case_insensitive_and_unknown_repo_is_empty() -> None:
    data = [_m("a.py", repo="X/One"), _m("b.py", repo="y/two")]
    only = compute_stats(data, repo="x/one")
    assert [r.repo for r in only.repos] == ["X/One"]
    assert only.total_chunks == 1
    assert compute_stats(data, repo="nope/none").repos == []


# --- import links ---------------------------------------------------------


def test_imports_between_files_are_found_and_outside_imports_counted() -> None:
    repo = compute_stats(SAMPLE).repos[0]
    edges = {(e.source, e.target) for e in repo.import_edges}
    assert edges == {
        ("src/pkg/a.py", "src/pkg/b.py"),   # pkg.b  (dotted)
        ("src/pkg/a.py", "src/pkg/c.py"),   # .c  (stored as "c", next to the file)
    }
    assert repo.import_edge_count == 2
    outside = {i.name: i.files for i in repo.top_external_imports}
    assert outside == {"os": 2, "json": 1, "requests": 1}


def test_a_file_that_imports_the_same_outside_name_twice_counts_once() -> None:
    data = [_m("a.py", imports="os,os.path")]
    assert {i.name: i.files for i in compute_stats(data).repos[0].top_external_imports} == {"os": 1}


def test_self_imports_are_not_links() -> None:
    data = [_m("pkg/a.py", imports="a")]
    repo = compute_stats(data).repos[0]
    assert repo.import_edges == []
    assert repo.top_external_imports == []


def test_ambiguous_names_are_skipped_not_guessed() -> None:
    data = [
        _m("one/util.py"), _m("two/util.py"),
        _m("main.py", imports="util"),
    ]
    repo = compute_stats(data).repos[0]
    assert repo.import_edges == []
    assert repo.top_external_imports == []


def test_a_package_import_points_at_its_init_file() -> None:
    data = [_m("src/pkg/__init__.py"), _m("src/tool.py", imports="pkg")]
    edges = [(e.source, e.target) for e in compute_stats(data).repos[0].import_edges]
    assert edges == [("src/tool.py", "src/pkg/__init__.py")]


def test_files_without_imports_data_give_no_links_and_no_crash() -> None:
    # Chunks saved by the old GitHub indexer had no imports at all.
    data = [{"qualified_name": "f", "file_path": "a.py"}]
    repo = compute_stats(data).repos[0]
    assert repo.import_edge_count == 0
    assert repo.chunks_by_type == {"unknown": 1}


def test_missing_file_path_is_reported_as_unknown() -> None:
    repo = compute_stats([{"qualified_name": "f"}]).repos[0]
    assert repo.chunks_per_file[0].file_path == "<unknown>"


@pytest.mark.parametrize("top", [1, 5, 50])
def test_import_edges_are_sorted_and_stable(top: int) -> None:
    data = [_m("a.py", imports="b,c"), _m("b.py"), _m("c.py")]
    repo = compute_stats(data, top=top).repos[0]
    assert [(e.source, e.target) for e in repo.import_edges] == [("a.py", "b.py"), ("a.py", "c.py")]
