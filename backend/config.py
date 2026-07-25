"""
Application configuration.

Centralized settings management using pydantic-settings.
All secrets and environment-specific values are loaded from a `.env` file
or process environment variables — never hard-coded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of the `backend/` directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application metadata ---
    app_name: str = Field(default="AI GitHub Agent", description="Human-readable app name.")
    app_version: str = Field(default="0.1.0", description="Semantic version of the backend.")
    environment: str = Field(default="development", description="dev / staging / production.")
    debug: bool = Field(default=True, description="Enable verbose error responses.")

    # --- API Keys (wrapped in SecretStr so they never accidentally print) ---
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key for LLM calls.",
    )
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Google Gemini API key (optional alternative LLM).",
    )
    github_token: SecretStr = Field(
        default=SecretStr(""),
        description="GitHub Personal Access Token for repo / PR / issue operations.",
    )

    # --- Database / storage paths ---
    vector_db_path: Path = Field(
        default=PROJECT_ROOT / "vector_db",
        description="Local filesystem path for the ChromaDB persistent store.",
    )
    repo_cache_path: Path = Field(
        default=PROJECT_ROOT / ".repo_cache",
        description="Directory where cloned GitHub repos are cached.",
    )

    # --- Server ---
    host: str = Field(default="127.0.0.1", description="Bind host for uvicorn.")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port for uvicorn.")

    def ensure_storage_dirs(self) -> None:
        """Create on-disk directories required by the app if they don't exist."""
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.repo_cache_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached `Settings` instance.

    Using `lru_cache` guarantees we instantiate (and parse `.env`) exactly once
    per process, which is both faster and FastAPI-dependency-friendly.
    """
    return Settings()