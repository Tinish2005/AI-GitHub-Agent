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
    admin_api_key: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "Secret that unlocks the indexing endpoints (send it as header X-Admin-Key). "
            "Empty means indexing is closed to everyone."
        ),
    )
    gemini_llm_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model used for answers, fixes and plans. Copy the exact model id from Google AI Studio.",
    )
    public_indexing_enabled: bool = Field(
        default=False,
        description="Let visitors index their own PUBLIC GitHub repo (within the limits below).",
    )
    public_index_max_files: int = Field(default=10, ge=1, le=50)
    public_index_max_chunks: int = Field(default=150, ge=1, le=2000)
    public_index_max_file_bytes: int = Field(default=200_000, ge=1_000)
    public_index_per_ip_per_hour: int = Field(default=1, ge=1)
    public_index_per_ip_per_day: int = Field(default=2, ge=1)
    public_index_attempts_per_ip_per_hour: int = Field(default=20, ge=1)
    public_index_embeddings_per_day: int = Field(
        default=400, ge=0,
        description="Embedding calls visitors may use per day (Gemini free tier allows 1,000 in total).",
    )
    public_index_store_max_chunks: int = Field(default=900, ge=1)
    ai_calls_per_day: int = Field(
        default=16, ge=0, description="AI calls per day for everyone (Gemini 2.5 Flash free tier allows 20).",
    )
    ai_calls_per_minute: int = Field(default=4, ge=1, description="Gemini 2.5 Flash free tier allows 5.")
    ai_calls_per_ip_per_hour: int = Field(default=6, ge=1)
    trusted_proxy_hops: int = Field(
        default=0, ge=0, le=5,
        description="How many proxies sit in front of the server. 0 means: ignore X-Forwarded-For.",
    )
    allow_regenerated_pr: bool = Field(
        default=False,
        description=(
            "Escape hatch. When true, /fix/pr may generate its own diff if the client sends no "
            "reviewed proposal. Leave false so a pull request never contains an unreviewed diff."
        ),
    )
    gemini_embedding_model: str = Field(
        default="",
        description=(
            "Pin the Gemini embedding model, e.g. gemini-embedding-001. Empty means: try "
            "the built-in list in order. Pinning stops a temporary error from switching models."
        ),
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
    demo_seed_path: Path = Field(
        default=PROJECT_ROOT / "demo_seed" / "psf_requests.json.gz",
        description="Compressed demo index that is restored at startup when the live store is empty.",
    )
    restore_demo_on_startup: bool = Field(
        default=True,
        description="Restore the demo index at startup if the live store is missing or empty.",
    )
    repo_cache_path: Path = Field(
        default=PROJECT_ROOT / ".repo_cache",
        description="Directory where cloned GitHub repos are cached.",
    )

    # --- CORS ---
    cors_origins: str = Field(
        default="*",
        description=(
            "Frontend addresses allowed to call this API, comma-separated, e.g. "
            "https://my-app.vercel.app. Empty or * allows every website (credentials are then off)."
        ),
    )
    cors_origin_regex: str = Field(
        default="",
        description=(
            "Optional pattern for addresses that change on every deploy, such as Vercel previews: "
            "^https://my-app(-[a-z0-9-]+)?\\.vercel\\.app$"
        ),
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
