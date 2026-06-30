"""
FastAPI application entry point for the AI GitHub Agent backend.

Run locally with:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from backend.config import Settings, get_settings
from backend.indexing.embeddings import EmbeddingService
from backend.indexing.vector_store import VectorStore
from backend.mcp.models import JsonRpcRequest, JsonRpcResponse
from backend.mcp.server import MCPServer
from backend.mcp.tools import build_default_registry
from backend.rag.llm import EchoLLMClient, LLMClient, OpenAILLMClient
from backend.rag.models import Answer
from backend.rag.pipeline import RAGPipeline
from backend.github.client import GitHubAPI, GitHubClient
from backend.github.models import GitHubFile, RepoCoord


class HealthResponse(BaseModel):
    """Schema returned by the /health endpoint."""

    status: str
    app_name: str
    version: str
    environment: str


class QARequest(BaseModel):
    """Body for POST /qa."""

    question: str = Field(min_length=1, description="The user question.")
    top_k: int = Field(default=5, ge=1, le=20, description="How many chunks to retrieve.")


def get_vector_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> VectorStore:
    """Return a VectorStore wired to the configured path."""
    return VectorStore(
        persist_directory=settings.vector_db_path,
        embedding_service=EmbeddingService(),
    )


def get_llm_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LLMClient:
    """Pick a real or fake LLM client based on whether an OpenAI key is set."""
    key = settings.openai_api_key.get_secret_value()
    if key:
        return OpenAILLMClient(api_key=key)
    return EchoLLMClient()


def get_rag_pipeline(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    llm: Annotated[LLMClient, Depends(get_llm_client)],
) -> RAGPipeline:
    """Build the RAG pipeline by composing the vector store and LLM."""
    return RAGPipeline(vector_store=vector_store, llm=llm)


def get_github_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> GitHubAPI:
    """Build a GitHubClient using the configured token (no token = unauth)."""
    token = settings.github_token.get_secret_value() or None
    return GitHubClient(token=token)


def get_mcp_server(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
    github: Annotated[GitHubAPI, Depends(get_github_client)],
) -> MCPServer:
    """Build an MCP server with the default tool registry + GitHub tools."""
    registry = build_default_registry(vector_store, pipeline, github=github)
    return MCPServer(registry=registry)


def create_app() -> FastAPI:
    """Application factory - keeps each test instance isolated."""
    settings = get_settings()
    settings.ensure_storage_dirs()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )

    @app.get("/", response_model=HealthResponse, tags=["meta"])
    def root(
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_name=settings.app_name,
            version=settings.app_version,
            environment=settings.environment,
        )

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_name=settings.app_name,
            version=settings.app_version,
            environment=settings.environment,
        )

    @app.post("/qa", response_model=Answer, tags=["rag"])
    def qa(
        body: QARequest,
        pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
    ) -> Answer:
        """Answer a natural-language question about the indexed codebase."""
        try:
            pipeline.top_k = body.top_k
            return pipeline.ask(body.question)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/mcp", response_model=JsonRpcResponse, tags=["mcp"])
    def mcp(
        body: dict,
        server: Annotated[MCPServer, Depends(get_mcp_server)],
    ) -> JsonRpcResponse:
        """JSON-RPC 2.0 endpoint exposing MCP tools to clients."""
        try:
            request = JsonRpcRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return server.handle(request)

    @app.get("/github/file", response_model=GitHubFile, tags=["github"])
    def github_file(
        owner: str,
        repo: str,
        path: str,
        github: Annotated[GitHubAPI, Depends(get_github_client)],
        ref: str | None = None,
    ) -> GitHubFile:
        """Fetch a single file from a GitHub repository."""
        try:
            return github.get_file(RepoCoord(owner=owner, repo=repo), path, ref=ref)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


# Module-level instance for `uvicorn backend.main:app`.
app: FastAPI = create_app()