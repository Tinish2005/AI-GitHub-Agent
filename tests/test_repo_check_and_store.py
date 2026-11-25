"""Loop 8 tests: the public-repo check, indexer limits, and store repo helpers (OFFLINE)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from backend.github.models import RepoCoord
from backend.github.repo_check import NOT_PUBLIC_MESSAGE, RepoCheckError, check_public_repo
from backend.github.repo_indexer import GitHubRepoIndexer
from backend.indexing.ast_parser import parse_source
from tests.test_index_safety import ALPHA, BETA, _store


# --- the public repo check ---------------------------------------------------------------


def transport(status: int = 200, json_body: dict | None = None, headers: dict | None = None, seen: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=json_body or {}, headers=headers or {})
    return httpx.MockTransport(handler)


def test_a_public_repo_passes_and_gets_its_proper_capitalisation() -> None:
    seen: list = []
    body = {"full_name": "Psf/Requests", "private": False, "visibility": "public", "default_branch": "main"}
    info = check_public_repo("psf", "requests", "tok", transport=transport(200, body, seen=seen))
    assert info == {"full_name": "Psf/Requests", "default_branch": "main"}
    assert seen[0].headers["authorization"] == "Bearer tok"
    assert seen[0].url.path == "/repos/psf/requests"


def test_a_public_repo_without_a_token_is_checked_without_authorization() -> None:
    seen: list = []
    check_public_repo("o", "r", None, transport=transport(200, {"private": False}, seen=seen))
    assert "authorization" not in seen[0].headers


@pytest.mark.parametrize("body", [
    {"private": True, "visibility": "private"},
    {"private": False, "visibility": "internal"},
    {"visibility": "private"},
])
def test_a_private_repo_is_refused_with_the_same_message_as_a_missing_one(body: dict) -> None:
    with pytest.raises(RepoCheckError) as private:
        check_public_repo("o", "r", "tok", transport=transport(200, body))
    with pytest.raises(RepoCheckError) as missing:
        check_public_repo("o", "r", "tok", transport=transport(404))
    assert private.value.message == missing.value.message == NOT_PUBLIC_MESSAGE
    assert private.value.status == missing.value.status == 404


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_lookup_does_not_reveal_anything(status: int) -> None:
    with pytest.raises(RepoCheckError) as exc:
        check_public_repo("o", "r", "tok", transport=transport(status, {"message": "Forbidden"}))
    assert exc.value.message == NOT_PUBLIC_MESSAGE


def test_a_github_rate_limit_is_reported_as_a_temporary_problem() -> None:
    with pytest.raises(RepoCheckError) as exc:
        check_public_repo("o", "r", None, transport=transport(403, {"message": "API rate limit exceeded"}, {"x-ratelimit-remaining": "0"}))
    assert exc.value.status == 503 and "limiting requests" in exc.value.message


def test_a_github_error_or_network_failure_is_a_temporary_problem() -> None:
    with pytest.raises(RepoCheckError) as exc:
        check_public_repo("o", "r", None, transport=transport(500))
    assert exc.value.status == 503

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(RepoCheckError) as net:
        check_public_repo("o", "r", None, transport=httpx.MockTransport(boom))
    assert net.value.status == 503


@pytest.mark.parametrize("owner,repo", [("../x", "r"), ("o", "r/../y"), ("o w", "r"), ("", "r"), ("o", "")])
def test_bad_names_never_reach_github(owner: str, repo: str) -> None:
    seen: list = []
    with pytest.raises(RepoCheckError) as exc:
        check_public_repo(owner, repo, None, transport=transport(200, {}, seen=seen))
    assert exc.value.status == 422 and seen == []


# --- indexer limits ---------------------------------------------------------------------------

BIG = "".join(f"def f{i}():\n    return {i}\n\n" for i in range(3))
FILES = {"src/a.py": ALPHA, "src/b.py": BETA, "src/c.py": BIG}


def indexer(tmp_path: Path, files: dict = FILES, fetch_fail: set | None = None, max_files: int = 10):
    store = _store(tmp_path / "vdb")
    ix = GitHubRepoIndexer(token=None, vector_store=store, max_files=max_files)
    ix._default_branch = lambda o, r: "main"
    ix._list_python_files = lambda o, r, ref: sorted(files)

    def fetch(o, r, path, ref):
        if fetch_fail and path in fetch_fail:
            raise RuntimeError("boom")
        return files[path]

    ix._fetch_file = fetch
    return ix, store


def test_without_limits_everything_is_indexed_and_the_new_fields_are_quiet(tmp_path: Path) -> None:
    ix, store = indexer(tmp_path)
    result = ix.index(RepoCoord(owner="o", repo="r"))
    assert (result.files_indexed, result.chunks_indexed, result.files_skipped, result.truncated) == (3, 5, 0, False)
    assert store.count() == 5


def test_the_chunk_budget_stops_adding_files_and_says_so(tmp_path: Path) -> None:
    ix, store = indexer(tmp_path)
    result = ix.index(RepoCoord(owner="o", repo="r"), max_chunks=2)
    assert (result.chunks_indexed, result.files_indexed, result.files_skipped, result.truncated) == (2, 2, 1, True)
    # Windows writes backslashes in file paths, Linux slashes: compare them the same way.
    assert {m["file_path"].replace("\\", "/") for m in store.all_metadata()} == {"src/a.py", "src/b.py"}


def test_a_file_over_the_byte_limit_is_skipped(tmp_path: Path) -> None:
    files = {"src/a.py": ALPHA, "src/big.py": ALPHA + "# " + "x" * 2000 + "\n"}
    ix, store = indexer(tmp_path, files)
    result = ix.index(RepoCoord(owner="o", repo="r"), max_file_bytes=1000)
    assert (result.files_indexed, result.files_skipped, result.truncated) == (1, 1, False)
    assert store.count() == 1


def test_a_file_that_cannot_be_fetched_is_counted_as_skipped(tmp_path: Path) -> None:
    ix, store = indexer(tmp_path, fetch_fail={"src/b.py"})
    result = ix.index(RepoCoord(owner="o", repo="r"))
    assert (result.files_indexed, result.files_skipped, result.chunks_indexed) == (2, 1, 4)


def test_progress_is_reported_while_fetching_and_before_embedding(tmp_path: Path) -> None:
    ix, _ = indexer(tmp_path)
    events: list = []
    ix.index(RepoCoord(owner="o", repo="r"), progress=lambda phase, done, total: events.append((phase, done, total)))
    assert events == [("fetching", 0, 3), ("fetching", 1, 3), ("fetching", 2, 3), ("fetching", 3, 3), ("embedding", 0, 5)]


def test_the_before_add_hook_sees_the_chunk_count_and_can_stop_the_run(tmp_path: Path) -> None:
    ix, store = indexer(tmp_path)
    counts: list = []
    ix.index(RepoCoord(owner="o", repo="r"), before_add=counts.append)
    assert counts == [5]

    ix2, store2 = indexer(tmp_path / "two")

    def refuse(count: int) -> None:
        raise RuntimeError("no budget")

    with pytest.raises(RuntimeError, match="no budget"):
        ix2.index(RepoCoord(owner="o", repo="r"), before_add=refuse)
    assert store2.count() == 0


def test_nothing_to_index_never_calls_the_hook(tmp_path: Path) -> None:
    ix, _ = indexer(tmp_path, files={"src/e.py": "x = 1\n"})
    called: list = []
    result = ix.index(RepoCoord(owner="o", repo="r"), before_add=called.append)
    assert result.chunks_indexed == 0 and called == []


# --- the store's repo helpers ----------------------------------------------------------------------


def labelled(store, source: str, name: str, repo: str, at: str, demo: bool = False) -> None:
    tags = {"repo": repo, "indexed_at": at}
    if demo:
        tags["demo"] = "true"
    store.add_chunks(parse_source(source, Path(name)), extra_metadata=tags)


def test_repo_summaries_count_chunks_flag_the_demo_and_report_the_newest_time(tmp_path: Path) -> None:
    store = _store(tmp_path)
    labelled(store, ALPHA + "\n" + BETA, "a.py", "o/one", "2026-01-01T00:00:00+00:00")
    labelled(store, "def late():\n    return 1\n", "b.py", "o/one", "2026-03-01T00:00:00+00:00")
    labelled(store, "def d():\n    return 1\n", "d.py", "demo/repo", "2026-02-01T00:00:00+00:00", demo=True)
    store.add_chunks(parse_source("def u():\n    return 1\n", Path("u.py")))   # no label
    summaries = store.repo_summaries()
    assert set(summaries) == {"o/one", "demo/repo"}
    assert summaries["o/one"] == {"chunks": 3, "demo": False, "indexed_at": "2026-03-01T00:00:00+00:00"}
    assert summaries["demo/repo"]["demo"] is True


def test_removing_a_repo_keeps_everything_else_and_survives_a_reload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    labelled(store, ALPHA, "a.py", "o/one", "2026-01-01T00:00:00+00:00")
    labelled(store, BETA, "b.py", "o/two", "2026-01-02T00:00:00+00:00")
    assert store.remove_repo("O/ONE") == 1
    reloaded = _store(tmp_path)
    assert reloaded.count() == 1 and set(reloaded.repo_summaries()) == {"o/two"}
    hits = reloaded.query("beta", top_k=3)
    assert [h.metadata["repo"] for h in hits] == ["o/two"]       # vectors still line up with their chunks
    assert reloaded.embedding_info()["dimension"] == 3


def test_removing_an_unknown_repo_changes_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    labelled(store, ALPHA, "a.py", "o/one", "2026-01-01T00:00:00+00:00")
    before = (tmp_path / "idx" / "store.json").read_bytes()
    assert store.remove_repo("nope/none") == 0
    assert (tmp_path / "idx" / "store.json").read_bytes() == before


def test_a_store_emptied_by_removal_can_be_used_again(tmp_path: Path) -> None:
    store = _store(tmp_path)
    labelled(store, ALPHA, "a.py", "o/one", "2026-01-01T00:00:00+00:00")
    assert store.remove_repo("o/one") == 1 and store.count() == 0
    assert store.query("alpha") == []
    labelled(store, BETA, "b.py", "o/two", "2026-01-02T00:00:00+00:00")
    assert store.count() == 1 and store.query("beta", repo="o/two")
