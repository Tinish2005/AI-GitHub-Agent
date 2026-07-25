"""Unit tests for `backend.config`."""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr

from backend.config import Settings, get_settings


def test_settings_defaults_are_sane() -> None:
    """A freshly-constructed Settings object should have safe defaults."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.app_name == "AI GitHub Agent"
    assert settings.app_version == "0.1.0"
    assert settings.environment == "development"
    assert settings.debug is True
    assert settings.host == "127.0.0.1"
    assert 1 <= settings.port <= 65535


def test_secret_keys_are_wrapped_in_secretstr() -> None:
    """API keys must be SecretStr so they don't leak via __repr__."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert isinstance(settings.openai_api_key, SecretStr)
    assert isinstance(settings.gemini_api_key, SecretStr)
    assert isinstance(settings.github_token, SecretStr)


def test_storage_paths_are_path_objects() -> None:
    """vector_db_path and repo_cache_path should always be Path instances."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert isinstance(settings.vector_db_path, Path)
    assert isinstance(settings.repo_cache_path, Path)


def test_ensure_storage_dirs_creates_directories(tmp_path: Path) -> None:
    """ensure_storage_dirs should create both directories idempotently."""
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        vector_db_path=tmp_path / "vdb",
        repo_cache_path=tmp_path / "repos",
    )

    settings.ensure_storage_dirs()
    assert settings.vector_db_path.is_dir()
    assert settings.repo_cache_path.is_dir()

    # Calling twice must not raise.
    settings.ensure_storage_dirs()
    assert settings.vector_db_path.is_dir()


def test_get_settings_is_cached() -> None:
    """get_settings should return the same instance on repeated calls."""
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b