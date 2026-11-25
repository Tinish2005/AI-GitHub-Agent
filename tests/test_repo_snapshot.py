"""Loop 3 tests: fetching real files from GitHub (OFFLINE, simulated GitHub).

These use a fake HTTP transport. They prove the request/response handling and
the safety rules. They do NOT prove the real GitHub API still behaves this way;
that is covered by scripts/live_check_validate.py.
"""

from __future__ import annotations

import httpx
import pytest

from backend.github.repo_snapshot import (
    GitHubSnapshotProvider,
    SnapshotError,
    safe_repo_path,
)

SHA = "a" * 40


def _tree(*entries, truncated: bool = False) -> dict:
    return {
        "sha": SHA,
        "truncated": truncated,
        "tree": [{"path": p, "type": "blob", "size": size} for p, size in entries],
    }


def _provider(handler, **kwargs) -> GitHubSnapshotProvider:
    return GitHubSnapshotProvider(transport=httpx.MockTransport(handler), **kwargs)


def _happy_handler(files: dict, seen: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        path = request.url.path
        if path.startswith("/repos/o/r/commits/"):
            return httpx.Response(200, json={"sha": SHA})
        if path == f"/repos/o/r/git/trees/{SHA}":
            return httpx.Response(200, json=_tree(*[(p, len(b)) for p, b in files.items()]))
        if path.startswith("/repos/o/r/contents/"):
            name = path[len("/repos/o/r/contents/"):]
            if name in files:
                return httpx.Response(200, content=files[name])
        return httpx.Response(404, json={"message": "Not Found"})
    return handler


FILES = {
    "src/pkg/a.py": b"x = 1\n",
    "src/pkg/__init__.py": b"",
    "README.md": b"hello\n",
}


def test_fetch_returns_commit_files_missing_and_python_paths() -> None:
    p = _provider(_happy_handler(FILES))
    snap = p.fetch("o", "r", None, ["src/pkg/a.py", "src/pkg/nope.py"])
    assert snap.commit == SHA
    assert snap.files == {"src/pkg/a.py": "x = 1\n"}
    assert snap.missing == ("src/pkg/nope.py",)
    assert snap.tree_paths == ("src/pkg/__init__.py", "src/pkg/a.py")


def test_fetch_uses_the_ref_and_then_pins_files_to_the_commit() -> None:
    seen: list = []
    p = _provider(_happy_handler(FILES, seen))
    p.fetch("o", "r", "feature/x", ["src/pkg/a.py"])
    assert seen[0].url.path == "/repos/o/r/commits/feature/x"
    contents = [r for r in seen if r.url.path.startswith("/repos/o/r/contents/")]
    assert contents[0].url.params["ref"] == SHA


def test_default_ref_is_head() -> None:
    seen: list = []
    _provider(_happy_handler(FILES, seen)).fetch("o", "r", None, [])
    assert seen[0].url.path == "/repos/o/r/commits/HEAD"


def test_token_is_sent_as_bearer_header_and_never_appears_in_errors() -> None:
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(401, json={"message": "Bad credentials"})

    p = _provider(handler)
    p._token = "ghp_secret_value"
    with pytest.raises(SnapshotError) as exc:
        p.fetch("o", "r", None, [])
    assert seen[0].headers["authorization"] == "Bearer ghp_secret_value"
    assert "ghp_secret_value" not in str(exc.value)


def test_unknown_repo_gives_a_clear_not_found_error() -> None:
    p = _provider(lambda request: httpx.Response(404, json={"message": "Not Found"}))
    with pytest.raises(SnapshotError, match="Not found on GitHub"):
        p.fetch("o", "r", None, ["a.py"])


def test_rate_limit_gives_a_clear_error() -> None:
    p = _provider(lambda request: httpx.Response(
        403, json={"message": "API rate limit exceeded"}, headers={"x-ratelimit-remaining": "0"},
    ))
    with pytest.raises(SnapshotError, match="rate limit"):
        p.fetch("o", "r", None, [])


def test_network_failure_gives_a_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(SnapshotError, match="Could not reach GitHub"):
        _provider(handler).fetch("o", "r", None, [])


def test_file_over_the_size_cap_is_refused_before_download() -> None:
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if "/commits/" in request.url.path:
            return httpx.Response(200, json={"sha": SHA})
        return httpx.Response(200, json=_tree(("big.py", 5_000_000)))

    with pytest.raises(SnapshotError, match="too large"):
        _provider(handler).fetch("o", "r", None, ["big.py"])
    assert not any("/contents/" in p for p in seen)


def test_too_many_files_is_refused_without_calling_github() -> None:
    seen: list = []
    p = _provider(_happy_handler(FILES, seen), max_files=2)
    with pytest.raises(SnapshotError, match="limit"):
        p.fetch("o", "r", None, ["a.py", "b.py", "c.py"])
    assert seen == []


@pytest.mark.parametrize("owner,repo", [("../x", "r"), ("o", "r/../../y"), ("o w", "r"), ("", "r")])
def test_bad_owner_or_repo_names_never_reach_github(owner: str, repo: str) -> None:
    seen: list = []
    with pytest.raises(SnapshotError, match="Invalid owner or repo"):
        _provider(_happy_handler(FILES, seen)).fetch(owner, repo, None, [])
    assert seen == []


@pytest.mark.parametrize("ref", ["a b", "x..y", "main?x=1", ""])
def test_bad_refs_are_refused(ref: str) -> None:
    if ref == "":
        return  # empty means the default branch
    with pytest.raises(SnapshotError, match="Invalid ref"):
        _provider(_happy_handler(FILES)).fetch("o", "r", ref, [])


def test_binary_file_is_refused() -> None:
    p = _provider(_happy_handler({"a.py": b"\xff\xfe\x00bad"}))
    with pytest.raises(SnapshotError, match="not a UTF-8"):
        p.fetch("o", "r", None, ["a.py"])


def test_truncated_tree_falls_back_to_asking_for_the_file() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/commits/" in path:
            return httpx.Response(200, json={"sha": SHA})
        if "/git/trees/" in path:
            return httpx.Response(200, json=_tree(truncated=True))
        if path.endswith("/deep/file.py"):
            return httpx.Response(200, content=b"y = 2\n")
        return httpx.Response(404, json={"message": "Not Found"})

    snap = _provider(handler).fetch("o", "r", None, ["deep/file.py", "deep/none.py"])
    assert snap.files == {"deep/file.py": "y = 2\n"}
    assert snap.missing == ("deep/none.py",)


@pytest.mark.parametrize("bad", [
    "../etc/passwd", "a/../../b.py", "/abs/path.py", "C:\\x.py", "C:/x.py",
    "a\\b.py", ".git/config", "", "   ", "a\x00b.py",
])
def test_unsafe_paths_are_refused(bad: str) -> None:
    with pytest.raises(SnapshotError):
        safe_repo_path(bad)


def test_paths_are_cleaned_up() -> None:
    assert safe_repo_path("./src//pkg/a.py") == "src/pkg/a.py"
    assert safe_repo_path("a.py") == "a.py"
