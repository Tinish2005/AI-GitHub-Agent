"""Data models for GitHub-side payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RepoCoord(BaseModel):
    """A (owner, repo) pair identifying a GitHub repository."""

    model_config = {"frozen": True}

    owner: str = Field(min_length=1, description="GitHub user or org login.")
    repo: str = Field(min_length=1, description="Repository name.")

    def slug(self) -> str:
        """Return the canonical 'owner/repo' string."""
        return f"{self.owner}/{self.repo}"


class GitHubFile(BaseModel):
    """A file fetched from a GitHub repo."""

    model_config = {"frozen": True}

    path: str = Field(min_length=1)
    sha: str = Field(min_length=1)
    size: int = Field(ge=0)
    content: str = Field(description="Decoded UTF-8 file contents.")
    encoding: str = Field(default="base64", min_length=1)


class Issue(BaseModel):
    """A GitHub issue."""

    model_config = {"frozen": True}

    number: int = Field(ge=1)
    title: str = Field(min_length=1)
    state: str = Field(min_length=1, description="open | closed")
    author: str = Field(min_length=1)
    body: str = Field(default="")
    url: str = Field(min_length=1)


class PullRequest(BaseModel):
    """A GitHub pull request."""

    model_config = {"frozen": True}

    number: int = Field(ge=1)
    title: str = Field(min_length=1)
    state: str = Field(min_length=1, description="open | closed | merged")
    author: str = Field(min_length=1)
    head: str = Field(min_length=1, description="Source branch.")
    base: str = Field(min_length=1, description="Target branch.")
    body: str = Field(default="")
    url: str = Field(min_length=1)
    merged: bool = Field(default=False)