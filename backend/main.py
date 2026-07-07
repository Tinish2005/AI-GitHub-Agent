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
from backend.agent.executor import make_default_executors
from backend.agent.engine import ExecutionEngine, ExecutionResult
from backend.agent.fix_generator import FixGenerator
from backend.agent.fix_models import FixProposal
from backend.agent.validation_pipeline import ValidationPipeline
from backend.agent.validation_models import ValidationResult


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str


class QARequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class IndexRequest(BaseModel):
    url: str = Field(min_length=1)
    force: bool = Field(default=False)


class PlanRequest(BaseModel):
    goal: str = Field(min_length=1)
    strategy: str = Field(default="rule_based")


class ExecuteRequest(BaseModel):
    goal: str = Field(min_length=1)
    strategy: str = Field(default="rule_based")
    abort_on_failure: bool = Field(default=True)


class FixProposeRequest(BaseModel):
    goal: str = Field(min_length=1)
    context: str = Field(default="")


class FixValidateRequest(BaseModel):
    """Body for POST /fix/validate."""

    goal: str = Field(min_length=1, description="The user goal / bug description.")
    context: str = Field(default="", description="Optional retrieved code context.")


def get_vector_store(settings: Annotated[Settings, Depends(get_settings)]) -> VectorStore:
    return VectorStore(
        persist_directory=settings.vector_db_path,
        embedding_service=EmbeddingService(),
    )


def get_llm_client(settings: Annotated[Settings, Depends(get_settings)]) -> LLMClient:
    key = settings.openai_api_key.get_secret_value()
    if key:
        return OpenAILLMClient(api_key=key)
    return EchoLLMClient()


def get_rag_pipeline(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    llm: Annotated[LLMClient, Depends(get_llm_client)],
) -> RAGPipeline:
    return RAGPipeline(vector_store=vector_store, llm=llm)


def get_github_client(settings: Annotated[Settings, Depends(get_settings)]) -> GitHubAPI:
    token = settings.github_token.get_secret_value() or None
    return GitHubClient(token=token)


def get_cloner(settings: Annotated[Settings, Depends(get_settings)]) -> Cloner:
    return GitCloner(cache_dir=settings.repo_cache_path)


def get_indexing_service(
    cloner: Annotated[Cloner, Depends(get_cloner)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> IndexingService:
    return IndexingService(cloner=cloner, vector_store=vector_store)


def get_planner(llm: Annotated[LLMClient, Depends(get_llm_client)]) -> Planner:
    return RuleBasedPlanner()


def get_fix_generator(llm: Annotated[LLMClient, Depends(get_llm_client)]) -> FixGenerator:
    return FixGenerator(llm=llm)


def get_validation_pipeline() -> ValidationPipeline:
    """Default: no base_root, uses default checks (syntax + imports + placeholders)."""
    return ValidationPipeline()


def get_execution_engine(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
    github: Annotated[GitHubAPI, Depends(get_github_client)],
    generator: Annotated[FixGenerator, Depends(get_fix_generator)],
    validator: Annotated[ValidationPipeline, Depends(get_validation_pipeline)],
) -> ExecutionEngine:
    executors = make_default_executors(
        vector_store, pipeline,
        github=github, generator=generator, validation_pipeline=validator,
    )
    return ExecutionEngine(executors=executors)


def get_mcp_server(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
    github: Annotated[GitHubAPI, Depends(get_github_client)],
    indexer: Annotated[IndexingService, Depends(get_indexing_service)],
    planner: Annotated[Planner, Depends(get_planner)],
    engine: Annotated[ExecutionEngine, Depends(get_execution_engine)],
    generator: Annotated[FixGenerator, Depends(get_fix_generator)],
    validator: Annotated[ValidationPipeline, Depends(get_validation_pipeline)],
) -> MCPServer:
    registry = build_default_registry(
        vector_store, pipeline,
        github=github, indexer=indexer, planner=planner,
        engine=engine, generator=generator, validator=validator,
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
        if body.strategy == "llm":
            planner: Planner = LLMPlanner(llm=llm)
        elif body.strategy == "rule_based":
            planner = RuleBasedPlanner()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {body.strategy}")
        try:
            return planner.plan(body.goal)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/execute", response_model=ExecutionResult, tags=["agent"])
    def execute_endpoint(
        body: ExecuteRequest,
        llm: Annotated[LLMClient, Depends(get_llm_client)],
        engine: Annotated[ExecutionEngine, Depends(get_execution_engine)],
    ) -> ExecutionResult:
        if body.strategy == "llm":
            planner: Planner = LLMPlanner(llm=llm)
        elif body.strategy == "rule_based":
            planner = RuleBasedPlanner()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {body.strategy}")
        try:
            plan = planner.plan(body.goal)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        engine.abort_on_failure = body.abort_on_failure
        return engine.run(plan)

    @app.post("/fix/propose", response_model=FixProposal, tags=["agent"])
    def fix_propose_endpoint(
        body: FixProposeRequest,
        generator: Annotated[FixGenerator, Depends(get_fix_generator)],
    ) -> FixProposal:
        try:
            return generator.propose(body.goal, body.context)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/fix/validate", response_model=ValidationResult, tags=["agent"])
    def fix_validate_endpoint(
        body: FixValidateRequest,
        generator: Annotated[FixGenerator, Depends(get_fix_generator)],
        validator: Annotated[ValidationPipeline, Depends(get_validation_pipeline)],
    ) -> ValidationResult:
        """Generate a fix + validate it in a sandbox. Returns per-check results."""
        try:
            proposal = generator.propose(body.goal, body.context)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return validator.validate(proposal)

    return app


app: FastAPI = create_app()