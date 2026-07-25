"""Unit tests for backend.cloning.cloner (FakeCloner + helpers only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.cloning.cloner import CloneResult, FakeCloner, GitCloner, repo_folder_name


def test_repo_folder_name_is_deterministic() -> None:
    name1 = repo_folder_name("https://github.com/o/r.git")
    name2 = repo_folder_name("https://github.com/o/r.git")
    assert name1 == name2


def test_repo_folder_name_includes_tail() -> None:
    name = repo_folder_name("https://github.com/Tinish2005/AI-GitHub-Agent.git")
    assert name.startswith("AI-GitHub-Agent-")


def test_repo_folder_name_rejects_empty() -> None:
    with pytest.raises(ValueError):
        repo_folder_name("")


def test_fake_cloner_creates_default_files(tmp_path: Path) -> None:
    cloner = FakeCloner(cache_dir=tmp_path)
    result = cloner.clone("https://github.com/o/r.git")
    assert isinstance(result, CloneResult)
    assert result.was_cached is False
    assert (result.local_path / "app.py").is_file()
    assert (result.local_path / "utils.py").is_file()


def test_fake_cloner_second_call_is_cached(tmp_path: Path) -> None:
    cloner = FakeCloner(cache_dir=tmp_path)
    r1 = cloner.clone("https://github.com/o/r.git")
    r2 = cloner.clone("https://github.com/o/r.git")
    assert r1.local_path == r2.local_path
    assert r2.was_cached is True


def test_fake_cloner_force_reclones(tmp_path: Path) -> None:
    cloner = FakeCloner(cache_dir=tmp_path)
    r1 = cloner.clone("https://github.com/o/r.git")
    r2 = cloner.clone("https://github.com/o/r.git", force=True)
    assert r2.was_cached is False
    assert r1.local_path == r2.local_path


def test_fake_cloner_supports_custom_files(tmp_path: Path) -> None:
    cloner = FakeCloner(
        cache_dir=tmp_path,
        files={"mymod.py": "def x():\n    return 1\n"},
    )
    result = cloner.clone("https://github.com/o/other.git")
    assert (result.local_path / "mymod.py").is_file()
    assert not (result.local_path / "app.py").exists()


def test_fake_cloner_rejects_empty_url(tmp_path: Path) -> None:
    cloner = FakeCloner(cache_dir=tmp_path)
    with pytest.raises(ValueError):
        cloner.clone("")


def test_git_cloner_rejects_empty_url(tmp_path: Path) -> None:
    cloner = GitCloner(cache_dir=tmp_path)
    with pytest.raises(ValueError):
        cloner.clone("")