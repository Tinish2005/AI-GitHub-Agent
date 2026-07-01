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
from backend.cloning.cloner import Cloner, GitCloner
from backend.cloning.indexing_service import IndexingService, IndexResult
from backend.agent.models import Plan
from backend.agent.planner import LLMPlanner, Planner, RuleBasedPlanner


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


class IndexRequest(BaseModel):
    """Body for POST /index."""

    url: str = Field(min_length=1, description="HTTPS GitHub URL to clone and index.")
    force: bool = Field(default=False, description="Re-clone even if cached.")


class PlanRequest(BaseModel):
    """Body for POST /plan."""

    goal: str = Field(min_length=1, description="The user's goal in plain English.")
    strategy: str = Field(default="rule_based", description="rule_based | llm")


def get_vector_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> VectorStore:
    return VectorStore(
        persist_directory=settings.vector_db_path,
        embedding_service=EmbeddingService(),
    )


def get_llm_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LLMClient:
    key = settings.openai_api_key.get_secret_value()
    if key:
        return OpenAILLMClient(api_key=key)
    return EchoLLMClient()


def get_rag_pipeline(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    llm: Annotated[LLMClient, Depends(get_llm_client)],
) -> RAGPipeline:
    return RAGPipeline(vector_store=vector_store, llm=llm)


def get_github_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> GitHubAPI:
    token = settings.github_token.get_secret_value() or None
    return GitHubClient(token=token)


def get_cloner(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Cloner:
    return GitCloner(cache_dir=settings.repo_cache_path)


def get_indexing_service(
    cloner: Annotated[Cloner, Depends(get_cloner)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> IndexingService:
    return IndexingService(cloner=cloner, vector_store=vector_store)


def get_planner(
    llm: Annotated[LLMClient, Depends(get_llm_client)],
) -> Planner:
    """Default planner - rule-based; /plan can override via 'strategy'."""
    return RuleBasedPlanner()


def get_mcp_server(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
    github: Annotated[GitHubAPI, Depends(get_github_client)],
    indexer: Annotated[IndexingService, Depends(get_indexing_service)],
    planner: Annotated[Planner, Depends(get_planner)],
) -> MCPServer:
    registry = build_default_registry(
        vector_store, pipeline, github=github, indexer=indexer, planner=planner,
    )
    return MCPServer(registry=registry)


def create_app() -> FastAPI:
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
        try:
            return github.get_file(RepoCoord(owner=owner, repo=repo), path, ref=ref)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/index", tags=["indexing"])
    def index_repo(
        body: IndexRequest,
        indexer: Annotated[IndexingService, Depends(get_indexing_service)],
    ) -> dict:
        try:
            result: IndexResult = indexer.index_repo(body.url, force=body.force)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "url": result.url,
            "local_path": str(result.local_path),
            "was_cached": result.was_cached,
            "files_scanned": result.files_scanned,
            "chunks_indexed": result.chunks_indexed,
        }

    @app.post("/plan", response_model=Plan, tags=["agent"])
    def plan_endpoint(
        body: PlanRequest,
        llm: Annotated[LLMClient, Depends(get_llm_client)],
    ) -> Plan:
        """Turn a user goal into a structured multi-step plan."""
        if body.strategy == "llm":
            planner: Planner = LLMPlanner(llm=llm)
        elif body.strategy == "rule_based":
            planner = RuleBasedPlanner()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy: {body.strategy}",
            )
        try:
            return planner.plan(body.goal)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


# Module-level instance for `uvicorn backend.main:app`.
app: FastAPI = create_app()