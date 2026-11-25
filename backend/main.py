"""FastAPI application entry point for the AI GitHub Agent backend."""

from __future__ import annotations

import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, model_validator
from backend.config import Settings, get_settings
from backend.cors import build_cors_config
from backend.indexing.embeddings import EmbeddingService
from backend.indexing.stats import StatsResponse, compute_stats
from backend.github.repo_check import check_public_repo
from backend.indexing.demo_seed import DemoSeedStatus, restore_demo_seed
from backend.indexing.jobs import IndexJobManager
from backend.limits import BudgetedLLM, LimitError, LimitGuard, client_ip
from backend.indexing.embedding_factory import build_embedding_service
from backend.indexing.vector_store import EmbeddingMismatchError, VectorStore
from backend.mcp.models import ErrorCode, JsonRpcError, JsonRpcRequest, JsonRpcResponse
from backend.mcp.server import MCPServer
from backend.mcp.tools import build_default_registry
from backend.rag.llm import EchoLLMClient, LLMClient, OpenAILLMClient
from backend.rag.gemini_llm import GeminiLLMClient
from backend.rag.models import Answer
from backend.rag.pipeline import RAGPipeline
from backend.github.client import GitHubAPI, GitHubClient
from backend.github.models import GitHubFile, RepoCoord
from backend.github.repo_indexer import GitHubRepoIndexer
from backend.cloning.cloner import Cloner, GitCloner
from backend.cloning.indexing_service import IndexingService, IndexResult
from backend.agent.models import Plan
from backend.agent.planner import LLMPlanner, Planner, RuleBasedPlanner
from backend.agent.executor import REPO_TARGET_KEY, make_default_executors
from backend.agent.repo_context import retrieve_repo_context
from backend.agent.engine import ExecutionEngine, ExecutionResult
from backend.agent.fix_generator import FixGenerator, proposal_from_reviewed
from backend.agent.fix_models import FixProposal, ReviewedDiffError, ReviewedProposal
from backend.agent.validation_pipeline import ValidationPipeline
from backend.agent.validation_models import RepoTarget
from backend.github.repo_snapshot import GitHubSnapshotProvider, SnapshotProvider
from backend.agent.validation_models import ValidationResult
from backend.agent.draft_pr_models import DraftPRRequest, DraftPRResult
from backend.agent.draft_pr_service import (
    DraftPRService, FakeDraftPRService, GitHubDraftPRService,
)


class HealthResponse(BaseModel):
    """Answer of GET/HEAD /health. Built from memory only: it never reads the index,
    never calls the AI and never calls GitHub, so it answers in milliseconds."""

    status: str = Field(
        description='"ok" means the server is up and ready. The server only starts answering '
        "after the demo restore step has finished, so any reply at all means it is awake.",
    )
    app_name: str
    version: str
    environment: str
    uptime_seconds: float = Field(
        default=0.0, description="Seconds since the server finished starting up.",
    )
    demo_restore: str = Field(
        default="unknown",
        description=(
            "What happened to the demo index at startup: restored (copied in just now), "
            "unchanged (a live index was already there, or the seed was refused), "
            "seed_missing (no seed file), disabled (switched off), failed (the restore step "
            "raised an error) or unknown (startup has not run). The server is ready in every case."
        ),
    )
    demo_message: str = Field(default="", description="Plain-language detail for demo_restore.")
    restore_seconds: float | None = Field(
        default=None, description="How long the demo restore took at startup.",
    )


def demo_restore_label(status: DemoSeedStatus | None, failed: bool = False) -> str:
    """One short word for the demo restore outcome (see HealthResponse.demo_restore)."""
    if failed:
        return "failed"
    if status is None:
        return "unknown"
    if not status.configured:
        return "disabled"
    if status.restored:
        return "restored"
    if not status.seed_found:
        return "seed_missing"
    return "unchanged"


def build_health(request: Request, settings: Settings) -> HealthResponse:
    """Everything here comes from memory. Do not add index, AI or GitHub calls to it."""
    state = request.app.state
    since = getattr(state, "ready_at", None) or getattr(state, "created_at", None)
    status = getattr(state, "demo_status", None)
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(time.monotonic() - since, 1) if since is not None else 0.0,
        demo_restore=demo_restore_label(status, bool(getattr(state, "demo_restore_failed", False))),
        demo_message=status.message if status is not None else "",
        restore_seconds=getattr(state, "restore_seconds", None),
    )


class _RepoFields(BaseModel):
    """Optional repository to work on. Give owner and repo together.

    With a repo, searches look only at that repo's chunks and fixes are
    checked against that repo.
    """

    owner: str | None = Field(default=None)
    repo: str | None = Field(default=None)
    ref: str | None = Field(default=None, description="Branch, tag or commit. Default branch if empty.")

    @model_validator(mode="after")
    def _owner_and_repo_together(self):
        if bool(self.owner) != bool(self.repo):
            raise ValueError("Give both owner and repo, or neither.")
        return self

    def repo_target(self) -> RepoTarget | None:
        if self.owner and self.repo:
            return RepoTarget(owner=self.owner, repo=self.repo, ref=self.ref or None)
        return None

    def scope(self) -> str | None:
        """"owner/name" used to limit searches, or None to search everything."""
        if self.owner and self.repo:
            return f"{self.owner}/{self.repo}"
        return None


class PublicIndexRequest(BaseModel):
    owner: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")
    repo: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")
    path: str = Field(default="", max_length=200, description="Optional folder to index, like src/mypackage.")


class QARequest(_RepoFields):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class IndexRequest(BaseModel):
    url: str = Field(min_length=1)
    force: bool = Field(default=False)


class GitHubIndexRequest(BaseModel):
    owner: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    max_files: int = Field(default=10, ge=1, le=50)


class PlanRequest(BaseModel):
    goal: str = Field(min_length=1)
    strategy: str = Field(default="rule_based")


class ExecuteRequest(_RepoFields):
    goal: str = Field(min_length=1)
    strategy: str = Field(default="rule_based")
    abort_on_failure: bool = Field(default=True)


class FixProposeRequest(_RepoFields):
    goal: str = Field(min_length=1)
    context: str = Field(default="")


class FixValidateRequest(_RepoFields):
    goal: str = Field(min_length=1)
    context: str = Field(default="")
    proposal: ReviewedProposal | None = Field(
        default=None,
        description="The proposal the user reviewed. If sent, this exact diff is validated.",
    )


class FixPRRequest(BaseModel):
    goal: str = Field(min_length=1)
    context: str = Field(default="")
    owner: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    base_branch: str = Field(default="main")
    proposal: ReviewedProposal | None = Field(
        default=None,
        description="The proposal the user reviewed. Required: the PR contains exactly this diff.",
    )


logger = logging.getLogger("backend.main")

# MCP tools that add data to the index. They need the admin key, like /index.
ADMIN_ONLY_TOOLS = frozenset({"index_repo"})
ADMIN_DENIED_MESSAGE = "Indexing needs the admin key (send it as the X-Admin-Key header)."
PR_NEEDS_REVIEWED_MESSAGE = (
    "Send the reviewed proposal (goal, explanation, diff) in the 'proposal' field. "
    "The server will not generate a new diff for a pull request."
)
REGENERATED_WARNING = (
    "No reviewed proposal was sent, so the server generated a new diff. "
    "It may differ from the one you saw."
)


def _is_admin(settings: Settings, supplied: str | None) -> bool:
    """True only when an admin key is configured and the supplied one matches it."""
    expected = settings.admin_api_key.get_secret_value()
    if not expected or not supplied:
        return False
    return secrets.compare_digest(expected.encode("utf-8"), supplied.encode("utf-8"))


def require_admin(
    settings: Annotated[Settings, Depends(get_settings)],
    x_admin_key: Annotated[str | None, Header()] = None,
) -> None:
    """Indexing is closed unless the caller sends the configured admin key."""
    if not _is_admin(settings, x_admin_key):
        if not settings.admin_api_key.get_secret_value():
            logger.warning("Indexing was refused: ADMIN_API_KEY is not set on this server.")
        raise HTTPException(status_code=403, detail=ADMIN_DENIED_MESSAGE)


def get_vector_store(settings: Annotated[Settings, Depends(get_settings)]) -> VectorStore:
    svc = build_embedding_service(settings)
    return VectorStore(persist_directory=settings.vector_db_path, embedding_service=svc)


def get_llm_client(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    x_admin_key: Annotated[str | None, Header()] = None,
) -> LLMClient:
    """The AI client. Real AI calls are counted against the demo's daily and hourly limits."""
    gemini_key = settings.gemini_api_key.get_secret_value()
    openai_key = settings.openai_api_key.get_secret_value()
    if gemini_key:
        inner: LLMClient = GeminiLLMClient(api_key=gemini_key, model=settings.gemini_llm_model)
    elif openai_key:
        inner = OpenAILLMClient(api_key=openai_key)
    else:
        return EchoLLMClient()  # free, so never counted
    guard = getattr(request.app.state, "guard", None)
    if guard is None:
        return inner
    return BudgetedLLM(
        inner, guard,
        ip=client_ip(request, settings.trusted_proxy_hops),
        is_admin=_is_admin(settings, x_admin_key),
    )


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


def get_fix_generator(llm: Annotated[LLMClient, Depends(get_llm_client)]) -> FixGenerator:
    return FixGenerator(llm=llm)


def get_snapshot_provider(settings: Annotated[Settings, Depends(get_settings)]) -> SnapshotProvider:
    token = settings.github_token.get_secret_value() or None
    return GitHubSnapshotProvider(token=token)


def get_validation_pipeline(
    provider: Annotated[SnapshotProvider, Depends(get_snapshot_provider)],
) -> ValidationPipeline:
    return ValidationPipeline(snapshot_provider=provider)


def get_draft_pr_service(settings: Annotated[Settings, Depends(get_settings)]) -> DraftPRService:
    token = settings.github_token.get_secret_value() or ""
    if token:
        return GitHubDraftPRService(token=token)
    return FakeDraftPRService()


def get_planner() -> Planner:
    return RuleBasedPlanner()


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
    draft_pr: Annotated[DraftPRService, Depends(get_draft_pr_service)],
) -> MCPServer:
    registry = build_default_registry(
        vector_store, pipeline,
        github=github, indexer=indexer, planner=planner,
        engine=engine, generator=generator, validator=validator,
        draft_pr_service=draft_pr,
    )
    return MCPServer(registry=registry)


def _retrieve_repo_context(vector_store, goal, top_k=5, repo=None):
    """Search the indexed repo for code relevant to the goal, return as text."""
    return retrieve_repo_context(vector_store, goal, top_k=top_k, repo=repo)


def create_app() -> FastAPI:
    settings = get_settings()
    settings.ensure_storage_dirs()
    cors = build_cors_config(settings.cors_origins, settings.cors_origin_regex)

    github_token = settings.github_token.get_secret_value() or None
    guard = LimitGuard(settings)
    jobs = IndexJobManager(
        guard=guard,
        settings=settings,
        store_factory=lambda: VectorStore(
            persist_directory=settings.vector_db_path,
            embedding_service=build_embedding_service(settings),
        ),
        indexer_factory=lambda store, max_files: GitHubRepoIndexer(
            token=github_token, vector_store=store, max_files=max_files,
        ),
        repo_checker=lambda owner, repo: check_public_repo(owner, repo, github_token),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Put the demo index in place when the live store is missing or empty.
        current = get_settings()
        restore_started = time.monotonic()
        restore_failed = False
        try:
            status = restore_demo_seed(
                current.demo_seed_path,
                current.vector_db_path,
                configured_model=current.gemini_embedding_model,
                enabled=current.restore_demo_on_startup,
            )
        except Exception as exc:  # never stop the app from starting
            restore_failed = True
            status = DemoSeedStatus(
                configured=True, message=f"The demo could not be restored ({type(exc).__name__}).",
            )
        app.state.demo_status = status
        app.state.demo_restore_failed = restore_failed
        app.state.restore_seconds = round(time.monotonic() - restore_started, 2)
        log = logger.warning if (status.configured and not status.restored and not status.seed_found) else logger.info
        log("Demo seed: %s", status.message)
        if status.warning:
            logger.warning("Demo seed: %s", status.warning)
        logger.info("Demo restore step took %.2f s.", app.state.restore_seconds)
        # The port opens after this point, so a /health reply means the server is ready.
        app.state.ready_at = time.monotonic()
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.state.guard = guard
    app.state.jobs = jobs
    app.state.created_at = time.monotonic()

    if cors.wildcard:
        (logger.warning if settings.environment.lower() == "production" else logger.info)(
            "CORS: every website may call this API. Set CORS_ORIGINS to your frontend address."
        )
    else:
        logger.info("CORS: allowed origins %s, pattern %s.", cors.allow_origins, cors.allow_origin_regex)
    app.add_middleware(CORSMiddleware, **cors.middleware_kwargs())

    @app.exception_handler(LimitError)
    async def limit_handler(request: Request, exc: LimitError):
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return JSONResponse(
            status_code=exc.status,
            content={"detail": exc.message, "retry_after_seconds": exc.retry_after},
            headers=headers,
        )

    @app.exception_handler(EmbeddingMismatchError)
    async def embedding_mismatch_handler(request: Request, exc: EmbeddingMismatchError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    # HEAD is allowed too, for uptime monitors.
    @app.head("/", include_in_schema=False)
    @app.get("/", response_model=HealthResponse, tags=["meta"])
    def root(request: Request, settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
        return build_health(request, settings)

    @app.head("/health", include_in_schema=False)
    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(request: Request, settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
        """Wake-up check. Answers from memory only; see HealthResponse for the fields."""
        return build_health(request, settings)

    @app.post("/qa", response_model=Answer, tags=["rag"])
    def qa(
        body: QARequest,
        pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
    ) -> Answer:
        try:
            pipeline.top_k = body.top_k
            scope = body.scope()
            return pipeline.ask(body.question, repo=scope) if scope else pipeline.ask(body.question)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/mcp", response_model=JsonRpcResponse, tags=["mcp"])
    def mcp(
        body: dict,
        server: Annotated[MCPServer, Depends(get_mcp_server)],
        settings: Annotated[Settings, Depends(get_settings)],
        x_admin_key: Annotated[str | None, Header()] = None,
    ) -> JsonRpcResponse:
        try:
            request = JsonRpcRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if request.method == "tools/call":
            tool_name = str(request.params.get("name", "")).strip()
            if tool_name in ADMIN_ONLY_TOOLS and not _is_admin(settings, x_admin_key):
                return JsonRpcResponse(
                    id=request.id,
                    error=JsonRpcError(code=ErrorCode.TOOL_EXECUTION_FAILED, message=ADMIN_DENIED_MESSAGE),
                )
        return server.handle(request)

    @app.get("/github/file", response_model=GitHubFile, tags=["github"])
    def github_file(
        owner: str, repo: str, path: str,
        github: Annotated[GitHubAPI, Depends(get_github_client)],
        ref: str | None = None,
    ) -> GitHubFile:
        try:
            return github.get_file(RepoCoord(owner=owner, repo=repo), path, ref=ref)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/index", tags=["indexing"], dependencies=[Depends(require_admin)])
    def index_repo(
        body: IndexRequest,
        request: Request,
        indexer: Annotated[IndexingService, Depends(get_indexing_service)],
    ) -> dict:
        try:
            with request.app.state.guard.admin_slot():
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

    @app.post("/index/github", tags=["indexing"], dependencies=[Depends(require_admin)])
    def index_github(
        body: GitHubIndexRequest,
        request: Request,
        settings: Annotated[Settings, Depends(get_settings)],
        vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    ) -> dict:
        token = settings.github_token.get_secret_value() or None
        indexer = GitHubRepoIndexer(
            token=token, vector_store=vector_store, max_files=body.max_files,
        )
        try:
            with request.app.state.guard.admin_slot():
                result = indexer.index(RepoCoord(owner=body.owner, repo=body.repo))
        except LimitError:
            raise
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "owner": result.owner,
            "repo": result.repo,
            "ref": result.ref,
            "files_indexed": result.files_indexed,
            "chunks_indexed": result.chunks_indexed,
        }

    @app.get("/stats", response_model=StatsResponse, tags=["indexing"])
    def stats_endpoint(
        request: Request,
        vector_store: Annotated[VectorStore, Depends(get_vector_store)],
        repo: Annotated[str | None, Query(description="Only this repo, as owner/name.")] = None,
        top: Annotated[int, Query(ge=1, le=200, description="Rows per list.")] = 20,
    ) -> StatsResponse:
        """Read-only statistics about what is indexed. No Gemini or GitHub calls."""
        try:
            metadatas = vector_store.all_metadata()
        except Exception as exc:
            raise HTTPException(status_code=500, detail="The index could not be read.") from exc
        stats = compute_stats(metadatas, repo=repo, top=top)
        info = getattr(vector_store, "embedding_info", None)
        if callable(info):
            details = info()
            stats.embedding_model = details.get("model")
            stats.embedding_dimension = details.get("dimension")
        stats.demo_seed = getattr(request.app.state, "demo_status", None)
        guard = getattr(request.app.state, "guard", None)
        stats.limits = guard.snapshot() if guard is not None else None
        return stats

    @app.post("/index/public", tags=["indexing"])
    def index_public(
        body: PublicIndexRequest,
        request: Request,
        settings: Annotated[Settings, Depends(get_settings)],
        x_admin_key: Annotated[str | None, Header()] = None,
    ) -> JSONResponse:
        """Start indexing a PUBLIC GitHub repo in the background. Takes a few minutes."""
        result = request.app.state.jobs.submit(
            body.owner, body.repo,
            ip=client_ip(request, settings.trusted_proxy_hops),
            is_admin=_is_admin(settings, x_admin_key),
            **({"path": body.path} if body.path.strip() else {}),
        )
        status = 200 if result.get("status") == "already_indexed" else 202
        return JSONResponse(status_code=status, content=result)

    @app.get("/index/public/{job_id}", tags=["indexing"])
    def index_public_status(job_id: str, request: Request) -> dict:
        job = request.app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such indexing job. It may be too old, or the server restarted.")
        return job

    @app.get("/admin/whoami", tags=["admin"], dependencies=[Depends(require_admin)])
    def whoami(
        request: Request,
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> dict:
        """Shows the address the server uses for rate limits. Use it to check TRUSTED_PROXY_HOPS."""
        return {
            "client_ip_used": client_ip(request, settings.trusted_proxy_hops),
            "direct_peer": request.client.host if request.client else None,
            "x_forwarded_for": request.headers.get("x-forwarded-for"),
            "trusted_proxy_hops": settings.trusted_proxy_hops,
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
        target = body.repo_target()
        initial = {REPO_TARGET_KEY: target} if target is not None else None
        return engine.run(plan, initial_outputs=initial)

    @app.post("/fix/propose", response_model=FixProposal, tags=["agent"])
    def fix_propose_endpoint(
        body: FixProposeRequest,
        generator: Annotated[FixGenerator, Depends(get_fix_generator)],
        vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    ) -> FixProposal:
        context = body.context or _retrieve_repo_context(vector_store, body.goal, repo=body.scope())
        try:
            return generator.propose(body.goal, context)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/fix/validate", response_model=ValidationResult, tags=["agent"])
    def fix_validate_endpoint(
        body: FixValidateRequest,
        generator: Annotated[FixGenerator, Depends(get_fix_generator)],
        validator: Annotated[ValidationPipeline, Depends(get_validation_pipeline)],
        vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    ) -> ValidationResult:
        if body.proposal is not None:
            try:
                proposal = proposal_from_reviewed(body.proposal)
            except ReviewedDiffError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            source, warning = "reviewed", ""
        else:
            context = body.context or _retrieve_repo_context(
                vector_store, body.goal, repo=body.scope(),
            )
            try:
                proposal = generator.propose(body.goal, context)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            source, warning = "regenerated", REGENERATED_WARNING
            logger.warning("/fix/validate generated its own diff (no reviewed proposal was sent).")
        result = validator.validate(proposal, repo=body.repo_target())
        return result.model_copy(update={"diff_source": source, "warning": warning})

    @app.post("/fix/pr", response_model=DraftPRResult, tags=["agent"])
    def fix_pr_endpoint(
        body: FixPRRequest,
        generator: Annotated[FixGenerator, Depends(get_fix_generator)],
        validator: Annotated[ValidationPipeline, Depends(get_validation_pipeline)],
        service: Annotated[DraftPRService, Depends(get_draft_pr_service)],
        vector_store: Annotated[VectorStore, Depends(get_vector_store)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> DraftPRResult:
        if body.proposal is not None:
            try:
                proposal = proposal_from_reviewed(body.proposal)
            except ReviewedDiffError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            source, warning = "reviewed", ""
        elif settings.allow_regenerated_pr:
            context = body.context or _retrieve_repo_context(
                vector_store, body.goal, repo=f"{body.owner}/{body.repo}",
            )
            try:
                proposal = generator.propose(body.goal, context)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            source, warning = "regenerated", REGENERATED_WARNING
            logger.warning("/fix/pr generated its own diff (ALLOW_REGENERATED_PR is on).")
        else:
            raise HTTPException(status_code=422, detail=PR_NEEDS_REVIEWED_MESSAGE)
        validation = validator.validate(
            proposal,
            repo=RepoTarget(owner=body.owner, repo=body.repo, ref=body.base_branch or None),
        )
        request = DraftPRRequest(
            owner=body.owner,
            repo=body.repo,
            base_branch=body.base_branch,
            goal=body.goal,
            proposal_explanation=proposal.explanation,
            proposal_diff=proposal.diff,
            confidence=proposal.confidence,
            validation_passed=validation.passed,
            validation_score=validation.score,
        )
        result = service.create(request)
        # The diff that was validated must be the diff that was submitted.
        assert validation.diff_sha256 == proposal.diff_sha256
        return result.model_copy(update={
            "diff_sha256": proposal.diff_sha256,
            "diff_source": source,
            "warning": warning,
        })

    return app


app: FastAPI = create_app()
