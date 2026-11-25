"""Loop 7 tests: the builder, restore and check scripts (OFFLINE, no Gemini, no GitHub)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from backend.github.repo_indexer import GitHubRepoIndexer
from backend.github.repo_snapshot import RepoSnapshot
from backend.indexing.demo_seed import DemoSeedError, export_seed
from tests.test_demo_seed import COMMIT, MODEL, _built, _load_seed, _write_seed
from tests.test_index_safety import ALPHA, BETA, _store

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- .gitignore check ----------------------------------------------------------

SEED = "demo_seed/psf_requests.json.gz"


@pytest.mark.parametrize("rules,ignored", [
    ("*.gz", True),
    ("demo_seed/", True),
    ("/demo_seed", True),
    ("demo_seed/*.gz", True),
    ("psf_requests.json.gz", True),
    ("*.json", False),
    ("vector_db/", False),
    ("build/\n*.pyc", False),
    ("# *.gz", False),
    ("", False),
    ("*.gz\n!psf_requests.json.gz", False),
    ("demo_seed/\n!demo_seed/psf_requests.json.gz", False),
    ("*.gz\n!other.gz", True),
    ("psf_requests.json.gz/", False),
    ("/psf_requests.json.gz", False),
])
def test_gitignore_matching(rules: str, ignored: bool) -> None:
    assert _load("check_demo_seed").is_ignored(SEED, rules) is ignored


# --- check_demo_seed -----------------------------------------------------------------


def _project(tmp: Path, gitignore: str | None = None) -> tuple:
    root = tmp / "project"
    seed = root / "demo_seed" / "psf_requests.json.gz"
    export_seed(_built(tmp), seed)
    if gitignore is not None:
        (root / ".gitignore").write_text(gitignore, encoding="utf-8")
    return root, seed


def _levels(results: list) -> list:
    return [level for level, _ in results]


def test_a_good_seed_in_a_clean_project_passes(tmp_path: Path) -> None:
    root, seed = _project(tmp_path, ".venv/\n__pycache__/\n")
    results = _load("check_demo_seed").check(seed, root, "gemini-embedding-001")
    assert "FAIL" not in _levels(results) and "WARN" not in _levels(results)
    assert any("Seed is valid: psf/requests" in m for _, m in results)


def test_a_missing_seed_fails_with_the_next_step(tmp_path: Path) -> None:
    results = _load("check_demo_seed").check(tmp_path / "nope.json.gz", tmp_path)
    assert _levels(results) == ["FAIL"] and "build_demo_seed" in results[0][1]


def test_a_damaged_seed_fails(tmp_path: Path) -> None:
    root, seed = _project(tmp_path)
    seed.write_bytes(b"garbage")
    assert "FAIL" in _levels(_load("check_demo_seed").check(seed, root, "x"))


def test_a_gitignore_that_blocks_the_seed_fails(tmp_path: Path) -> None:
    root, seed = _project(tmp_path, "*.gz\n")
    results = _load("check_demo_seed").check(seed, root, "gemini-embedding-001")
    assert any(level == "FAIL" and ".gitignore blocks" in m for level, m in results)


def test_an_unset_or_different_model_is_a_warning_not_a_failure(tmp_path: Path) -> None:
    root, seed = _project(tmp_path)
    check = _load("check_demo_seed").check
    unset = check(seed, root, "")
    assert "FAIL" not in _levels(unset) and any("not set" in m for l, m in unset if l == "WARN")
    other = check(seed, root, "gemini-embedding-002")
    assert "FAIL" not in _levels(other) and any("was built with" in m for l, m in other if l == "WARN")


def test_a_seed_outside_the_project_is_a_warning(tmp_path: Path) -> None:
    root, seed = _project(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    results = _load("check_demo_seed").check(seed, elsewhere, MODEL)
    assert any(l == "WARN" and "outside the project" in m for l, m in results)


def test_a_huge_seed_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, seed = _project(tmp_path)
    module = _load("check_demo_seed")
    monkeypatch.setattr(module, "MAX_MB", 0.0)
    assert any(l == "FAIL" and "100 MB" in m for l, m in module.check(seed, root, MODEL))


# --- build_demo_seed ---------------------------------------------------------------------


class _Provider:
    def __init__(self) -> None:
        self.calls: list = []

    def fetch(self, owner, repo, ref, paths):
        self.calls.append((owner, repo, ref, list(paths)))
        return RepoSnapshot(commit=COMMIT, files={}, missing=(), tree_paths=())


FILES = {
    "docs/conf.py": ALPHA,
    "examples/demo.py": ALPHA,
    "src/requests/a.py": ALPHA,
    "src/requests/b.py": BETA,
    "src/requests/c.py": "def gamma_fn():\n    return 'gamma'\n",
    "tests/test_x.py": BETA,
}


def _indexer(store, monkeypatch: pytest.MonkeyPatch, max_files: int = 50):
    indexer = GitHubRepoIndexer(token=None, vector_store=store, max_files=max_files)
    seen: list = []
    monkeypatch.setattr(indexer, "_default_branch", lambda o, r: "main")
    monkeypatch.setattr(indexer, "_list_python_files", lambda o, r, ref: sorted(FILES))
    monkeypatch.setattr(
        indexer, "_fetch_file", lambda o, r, p, ref: (seen.append((p, ref)), FILES[p])[1],
    )
    return indexer, seen


def test_the_builder_pins_one_commit_and_keeps_only_the_chosen_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = _load("build_demo_seed").build
    store = _store(tmp_path / "tmp", dim=6, model=MODEL)
    indexer, seen = _indexer(store, monkeypatch)
    provider = _Provider()
    out = tmp_path / "out" / "psf_requests.json.gz"
    manifest = build("psf", "requests", "src/", out, store=store, indexer=indexer, provider=provider)

    assert provider.calls == [("psf", "requests", None, [])]
    assert {p for p, _ in seen} == {"src/requests/a.py", "src/requests/b.py", "src/requests/c.py"}
    assert {ref for _, ref in seen} == {COMMIT}
    assert manifest["commit"] == COMMIT and manifest["branch"] == "main"
    assert manifest["include"] == "src/" and manifest["chunks"] == 3 and manifest["files"] == 3

    seed = _load_seed(out)
    assert {m["repo"] for m in seed["metadatas"]} == {"psf/requests"}
    assert {m["demo"] for m in seed["metadatas"]} == {"true"}
    assert {m["commit"] for m in seed["metadatas"]} == {COMMIT}
    assert {m["ref"] for m in seed["metadatas"]} == {"main"}
    assert not any("docs" in m["file_path"] or "tests" in m["file_path"] for m in seed["metadatas"])


def test_the_builder_respects_the_file_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build = _load("build_demo_seed").build
    store = _store(tmp_path / "tmp", dim=6, model=MODEL)
    indexer, seen = _indexer(store, monkeypatch, max_files=2)
    manifest = build("psf", "requests", "src/", tmp_path / "o.json.gz",
                     store=store, indexer=indexer, provider=_Provider())
    assert manifest["files"] == 2 and len(seen) == 2


def test_the_builder_fails_clearly_when_nothing_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build = _load("build_demo_seed").build
    store = _store(tmp_path / "tmp", dim=6, model=MODEL)
    indexer, _ = _indexer(store, monkeypatch)
    out = tmp_path / "o.json.gz"
    with pytest.raises(DemoSeedError, match="No chunks were indexed"):
        build("psf", "requests", "lib/", out, store=store, indexer=indexer, provider=_Provider())
    assert not out.exists()


def test_a_seed_made_by_the_builder_restores_and_passes_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.indexing.demo_seed import restore_demo_seed

    root = tmp_path / "project"
    out = root / "demo_seed" / "psf_requests.json.gz"
    store = _store(tmp_path / "tmp", dim=6, model=MODEL)
    indexer, _ = _indexer(store, monkeypatch)
    _load("build_demo_seed").build("psf", "requests", "src/", out,
                                   store=store, indexer=indexer, provider=_Provider())
    results = _load("check_demo_seed").check(out, root, "gemini-embedding-001")
    assert "FAIL" not in _levels(results)
    status = restore_demo_seed(out, tmp_path / "live")
    assert status.restored is True and status.chunks == 3


# --- restore_demo_seed script ----------------------------------------------------------------


def test_the_restore_script_restores_then_refuses_to_touch_the_result(tmp_path: Path, capsys) -> None:
    seed = tmp_path / "seed.json.gz"
    export_seed(_built(tmp_path), seed)
    module = _load("restore_demo_seed")
    args = ["restore", "--target", str(tmp_path / "live"), "--seed", str(seed)]
    assert module.main(args) == 0
    assert "Demo restored: psf/requests" in capsys.readouterr().out
    assert module.main(args) == 1
    assert "already has data" in capsys.readouterr().out


def test_the_restore_script_reports_a_missing_seed(tmp_path: Path, capsys) -> None:
    module = _load("restore_demo_seed")
    code = module.main(["restore", "--target", str(tmp_path / "live"), "--seed", str(tmp_path / "no.gz")])
    assert code == 1 and "No demo seed file found" in capsys.readouterr().out


def test_the_check_script_reports_a_manifest_next_to_the_seed(tmp_path: Path) -> None:
    root, seed = _project(tmp_path)
    manifest = json.loads((seed.parent / "psf_requests.manifest.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == COMMIT and manifest["dimension"] == 6
