"""Loop 7a tests: protect the index (OFFLINE, fake embeddings, no network).

Covers: unique chunk ids, no silent wipes, embedding model pinning and
checks, repo-scoped search, and the clear 503 error on a model mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.agent.engine import ExecutionEngine
from backend.agent.executor import make_default_executors
from backend.agent.fix_generator import FixGenerator
from backend.agent.fix_models import FixProposal
from backend.agent.draft_pr_service import FakeDraftPRService
from backend.agent.repo_context import retrieve_repo_context
from backend.config import Settings, get_settings
from backend.indexing.ast_parser import parse_source
from backend.indexing.embedding_factory import build_embedding_service
from backend.indexing.embeddings import EmbeddingService
from backend.indexing.vector_store import EmbeddingMismatchError, VectorStore
from backend.main import (
    app,
    get_draft_pr_service,
    get_execution_engine,
    get_fix_generator,
    get_llm_client,
    get_rag_pipeline,
    get_vector_store,
)
from backend.mcp.tools import make_ask_codebase_tool, make_search_code_tool
from backend.rag.gemini_embeddings import CANDIDATE_MODELS, GeminiEmbeddingBackend
from backend.rag.llm import EchoLLMClient
from backend.rag.pipeline import RAGPipeline


class _Backend:
    """Fake embeddings. `dim` and `model` can differ to simulate another model."""

    def __init__(self, dim: int = 3, model: str | None = None) -> None:
        self.dim = dim
        if model:
            self.model_name = model

    def encode(self, sentences: list, **kwargs) -> list:
        out = []
        for s in sentences:
            vec = [float(s.count("alpha")), float(s.count("beta")), 0.1]
            vec += [0.01 * (i + 1) for i in range(max(self.dim - 3, 0))]
            out.append(vec[: self.dim])
        return out


def _store(path: Path, *, dim: int = 3, model: str | None = None, name: str = "idx") -> VectorStore:
    return VectorStore(
        persist_directory=path,
        collection_name=name,
        embedding_service=EmbeddingService(backend=_Backend(dim, model)),
    )


ALPHA = "def alpha_fn():\n    return 'alpha'\n"
BETA = "def beta_fn():\n    return 'beta'\n"


def _two_repos(store: VectorStore) -> None:
    store.add_chunks(parse_source(ALPHA, Path("a.py")), extra_metadata={"repo": "o/a"})
    store.add_chunks(parse_source(BETA, Path("b.py")), extra_metadata={"repo": "o/b"})


def _state_file(path: Path, name: str = "idx") -> Path:
    return path / name / "store.json"


# --- unique chunk ids ------------------------------------------------------

TWINS = (
    "class One:\n    def __eq__(self, other):\n        return all(1 for _ in other)\n\n"
    "class Two:\n    def __eq__(self, other):\n        return all(1 for _ in other)\n"
)
OVERLOADS = (
    "def f(a: int) -> int: ...\n\n"
    "def f(a: str) -> str: ...\n\n"
    "def f(a):\n    return a\n"
)


def test_identical_code_in_two_places_is_kept_separately(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunks = parse_source(TWINS, Path("m.py"))
    ids = store.add_chunks(chunks, extra_metadata={"repo": "o/r"})
    names = {m["qualified_name"] for m in store.all_metadata()}
    assert len(ids) == len(set(ids)) == store.count() == len(chunks)
    assert {"One.__eq__", "Two.__eq__"} <= names


def test_functions_with_the_same_name_in_one_file_get_different_ids(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunks = parse_source(OVERLOADS, Path("m.py"))
    assert len({c.qualified_name for c in chunks}) == 1 and len(chunks) == 3
    ids = store.add_chunks(chunks)
    assert len(set(ids)) == 3 == store.count()


def test_indexing_again_changes_nothing_and_keeps_every_label(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunks = parse_source(TWINS, Path("m.py")) + parse_source(OVERLOADS, Path("n.py"))
    first = store.add_chunks(chunks, extra_metadata={"repo": "o/r"})
    second = store.add_chunks(chunks, extra_metadata={"repo": "o/r"})
    assert first == second
    assert store.count() == len(chunks)
    assert all(m.get("repo") == "o/r" for m in store.all_metadata())


def test_a_changed_function_replaces_its_old_version(tmp_path: Path) -> None:
    store = _store(tmp_path)
    label = {"repo": "o/r"}
    store.add_chunks(parse_source("def f():\n    return 1\n", Path("m.py")), extra_metadata=label)
    store.add_chunks(parse_source("def f():\n    return 2\n", Path("m.py")), extra_metadata=label)
    assert store.count() == 1
    assert store.query("f", top_k=1)[0].document.endswith("return 2")


def test_same_code_in_two_repos_stays_separate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunks = parse_source(ALPHA, Path("a.py"))
    store.add_chunks(chunks, extra_metadata={"repo": "o/one"})
    store.add_chunks(chunks, extra_metadata={"repo": "o/two"})
    assert store.count() == 2


def test_ids_do_not_depend_on_windows_or_posix_path_style(tmp_path: Path) -> None:
    a = _store(tmp_path / "a").add_chunks(parse_source(ALPHA, Path("src\\a.py")), extra_metadata={"repo": "o/r"})
    b = _store(tmp_path / "b").add_chunks(parse_source(ALPHA, Path("src/a.py")), extra_metadata={"repo": "o/r"})
    assert a == b


def test_repo_label_in_an_id_is_not_case_sensitive(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunks = parse_source(ALPHA, Path("a.py"))
    store.add_chunks(chunks, extra_metadata={"repo": "Psf/Requests"})
    store.add_chunks(chunks, extra_metadata={"repo": "psf/requests"})
    assert store.count() == 1


# --- the index is never wiped -----------------------------------------------


def _seed_index(tmp_path: Path) -> bytes:
    store = _store(tmp_path, dim=8, model="model-one")
    _two_repos(store)
    return _state_file(tmp_path).read_bytes()


def test_a_different_vector_size_is_refused_and_nothing_changes(tmp_path: Path) -> None:
    before = _seed_index(tmp_path)
    other = _store(tmp_path, dim=5, model="model-one")
    with pytest.raises(EmbeddingMismatchError, match="Nothing was changed"):
        other.add_chunks(parse_source(ALPHA, Path("c.py")), extra_metadata={"repo": "o/c"})
    assert _state_file(tmp_path).read_bytes() == before
    assert _store(tmp_path, dim=8, model="model-one").count() == 2


def test_a_different_model_with_the_same_size_is_refused(tmp_path: Path) -> None:
    before = _seed_index(tmp_path)
    other = _store(tmp_path, dim=8, model="model-two")
    with pytest.raises(EmbeddingMismatchError) as exc:
        other.add_chunks(parse_source(ALPHA, Path("c.py")))
    assert "model-one" in str(exc.value) and "model-two" in str(exc.value)
    assert _state_file(tmp_path).read_bytes() == before


def test_searching_with_the_wrong_model_or_size_says_so(tmp_path: Path) -> None:
    _seed_index(tmp_path)
    with pytest.raises(EmbeddingMismatchError):
        _store(tmp_path, dim=5, model="model-one").query("alpha")
    with pytest.raises(EmbeddingMismatchError):
        _store(tmp_path, dim=8, model="model-two").query("alpha")
    assert _store(tmp_path, dim=8, model="model-one").query("alpha")


def test_a_fresh_index_records_its_model_and_size_and_they_survive_a_reload(tmp_path: Path) -> None:
    store = _store(tmp_path, dim=8, model="model-one")
    assert store.embedding_info() == {"model": None, "dimension": None}
    _two_repos(store)
    assert _store(tmp_path, dim=8, model="model-one").embedding_info() == {
        "model": "model-one", "dimension": 8,
    }


def test_an_old_index_without_a_recorded_model_is_still_protected_by_size(tmp_path: Path) -> None:
    _seed_index(tmp_path)
    state = json.loads(_state_file(tmp_path).read_text(encoding="utf-8"))
    state.pop("embedding")
    _state_file(tmp_path).write_text(json.dumps(state), encoding="utf-8")
    same_size = _store(tmp_path, dim=8, model="other-name")
    same_size.add_chunks(parse_source(ALPHA, Path("c.py")))
    assert same_size.embedding_info()["model"] is None
    with pytest.raises(EmbeddingMismatchError):
        _store(tmp_path, dim=4).add_chunks(parse_source(ALPHA, Path("d.py")))


def test_reset_is_still_possible_on_purpose_and_forgets_the_model(tmp_path: Path) -> None:
    _seed_index(tmp_path)
    store = _store(tmp_path, dim=5, model="model-two")
    store.reset()
    store.add_chunks(parse_source(ALPHA, Path("a.py")))
    assert store.embedding_info() == {"model": "model-two", "dimension": 5}


def test_a_failed_save_leaves_the_old_file_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = _seed_index(tmp_path)
    store = _store(tmp_path, dim=8, model="model-one")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("backend.indexing.vector_store.json.dumps", boom)
    with pytest.raises(OSError):
        store.add_chunks(parse_source(ALPHA, Path("c.py")))
    assert _state_file(tmp_path).read_bytes() == before


def test_no_temporary_file_is_left_behind(tmp_path: Path) -> None:
    _seed_index(tmp_path)
    assert [p.name for p in (tmp_path / "idx").iterdir()] == ["store.json"]


# --- repo-scoped search -------------------------------------------------------


def test_a_scoped_search_only_returns_that_repos_chunks(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _two_repos(store)
    unscoped = store.query("alpha", top_k=5)
    assert unscoped[0].metadata["repo"] == "o/a" and len(unscoped) == 2
    scoped = store.query("alpha", top_k=5, repo="o/b")
    assert [r.metadata["repo"] for r in scoped] == ["o/b"]


def test_scope_is_case_insensitive_and_blank_means_everything(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _two_repos(store)
    assert len(store.query("alpha", repo="O/A")) == 1
    assert len(store.query("alpha", repo="   ")) == 2


def test_unknown_repo_gives_no_results(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _two_repos(store)
    assert store.query("alpha", repo="nope/none") == []


def test_chunks_without_a_label_never_match_a_scoped_search(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_chunks(parse_source(ALPHA, Path("old.py")))
    assert store.query("alpha", repo="o/a") == []
    assert len(store.query("alpha")) == 1


def test_scoped_search_with_fewer_chunks_than_top_k(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _two_repos(store)
    assert len(store.query("beta", top_k=20, repo="o/b")) == 1


def test_scoped_search_keeps_the_closest_first(tmp_path: Path) -> None:
    store = _store(tmp_path)
    src = "def one():\n    return 'alpha alpha'\n\ndef two():\n    return 'beta'\n"
    store.add_chunks(parse_source(src, Path("m.py")), extra_metadata={"repo": "o/a"})
    store.add_chunks(parse_source(BETA, Path("b.py")), extra_metadata={"repo": "o/b"})
    results = store.query("alpha", top_k=5, repo="o/a")
    assert [r.metadata["qualified_name"] for r in results] == ["one", "two"]
    assert results[0].distance <= results[1].distance


def test_context_helper_is_scoped_and_never_hides_a_mismatch(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _two_repos(store)
    text = retrieve_repo_context(store, "alpha", repo="o/b")
    assert "beta_fn" in text and "alpha_fn" not in text
    assert "alpha_fn" in retrieve_repo_context(store, "alpha")

    class _Mismatch:
        def query(self, *args, **kwargs):
            raise EmbeddingMismatchError("different model")

    class _Broken:
        def query(self, *args, **kwargs):
            raise RuntimeError("boom")

    with pytest.raises(EmbeddingMismatchError):
        retrieve_repo_context(_Mismatch(), "goal")
    assert retrieve_repo_context(_Broken(), "goal") == ""


def test_pipeline_answers_from_the_chosen_repo_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _two_repos(store)
    rag = RAGPipeline(vector_store=store, llm=EchoLLMClient())
    assert {s.file_path for s in rag.ask("alpha", repo="o/b").sources} == {"b.py"}
    assert {s.file_path for s in rag.ask("alpha").sources} == {"a.py", "b.py"}


def test_mcp_tools_accept_a_repo(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _two_repos(store)
    search = make_search_code_tool(store)
    assert "beta_fn" in search.execute({"query": "alpha", "repo": "o/b"})
    assert "alpha_fn" not in search.execute({"query": "alpha", "repo": "o/b"})
    assert "Found 2 matches" in search.execute({"query": "alpha"})
    ask = make_ask_codebase_tool(RAGPipeline(vector_store=store, llm=EchoLLMClient()))
    scoped = ask.execute({"question": "alpha", "repo": "o/b"})
    assert "b.py" in scoped and "a.py" not in scoped


# --- routes -----------------------------------------------------------------------


class _SpyGenerator:
    model = "spy"

    def __init__(self) -> None:
        self.contexts: list = []

    def propose(self, goal: str, context: str) -> FixProposal:
        self.contexts.append(context)
        return FixProposal(goal=goal, explanation="e", diff="d", model="spy", confidence=0.5)


@pytest.fixture
def api(tmp_path: Path) -> Iterator:
    store = _store(tmp_path)
    _two_repos(store)
    rag = RAGPipeline(vector_store=store, llm=EchoLLMClient())
    spy = _SpyGenerator()
    executors = make_default_executors(store, rag, generator=spy)  # type: ignore[arg-type]
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_rag_pipeline] = lambda: rag
    app.dependency_overrides[get_llm_client] = lambda: EchoLLMClient()
    app.dependency_overrides[get_fix_generator] = lambda: spy
    app.dependency_overrides[get_execution_engine] = lambda: ExecutionEngine(executors=executors)
    app.dependency_overrides[get_draft_pr_service] = lambda: FakeDraftPRService()
    yield SimpleNamespace(client=TestClient(app), spy=spy, store=store)
    app.dependency_overrides.clear()


def test_qa_route_is_scoped_when_a_repo_is_given(api) -> None:
    scoped = api.client.post("/qa", json={"question": "alpha", "owner": "o", "repo": "b"}).json()
    assert {s["file_path"] for s in scoped["sources"]} == {"b.py"}
    everything = api.client.post("/qa", json={"question": "alpha"}).json()
    assert {s["file_path"] for s in everything["sources"]} == {"a.py", "b.py"}


@pytest.mark.parametrize("payload", [{"owner": "o"}, {"repo": "b"}])
def test_qa_route_needs_owner_and_repo_together(api, payload: dict) -> None:
    assert api.client.post("/qa", json={"question": "alpha", **payload}).status_code == 422


def test_fix_propose_route_is_scoped(api) -> None:
    api.client.post("/fix/propose", json={"goal": "alpha", "owner": "o", "repo": "b"})
    assert "beta_fn" in api.spy.contexts[-1] and "alpha_fn" not in api.spy.contexts[-1]
    api.client.post("/fix/propose", json={"goal": "alpha"})
    assert "alpha_fn" in api.spy.contexts[-1]


def test_fix_pr_route_searches_only_the_repo_in_the_request(api) -> None:
    # This test sends only a goal, so it uses the escape hatch on purpose.
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, allow_regenerated_pr=True)
    api.client.post("/fix/pr", json={"goal": "alpha", "owner": "o", "repo": "b"})
    assert "beta_fn" in api.spy.contexts[-1] and "alpha_fn" not in api.spy.contexts[-1]


def test_fix_validate_route_is_scoped(api) -> None:
    api.client.post("/fix/validate", json={"goal": "alpha", "owner": "o", "repo": "b"})
    assert "beta_fn" in api.spy.contexts[-1] and "alpha_fn" not in api.spy.contexts[-1]


def test_execute_route_is_scoped_for_every_step(api) -> None:
    body = api.client.post("/execute", json={"goal": "fix the alpha bug", "owner": "o", "repo": "b"}).json()
    assert body["steps"]
    assert "beta_fn" in api.spy.contexts[-1] and "alpha_fn" not in api.spy.contexts[-1]
    analyze = next(s for s in body["steps"] if s["kind"] == "analyze")
    assert "b.py" in analyze["output"] and "a.py" not in analyze["output"]


def test_execute_route_without_a_repo_still_searches_everything(api) -> None:
    api.client.post("/execute", json={"goal": "fix the alpha bug"})
    assert "alpha_fn" in api.spy.contexts[-1] and "beta_fn" in api.spy.contexts[-1]


def test_a_model_mismatch_gives_a_clear_503_not_an_empty_answer(tmp_path: Path) -> None:
    _seed_index(tmp_path)
    wrong = _store(tmp_path, dim=8, model="model-two")
    rag = RAGPipeline(vector_store=wrong, llm=EchoLLMClient())
    app.dependency_overrides[get_rag_pipeline] = lambda: rag
    try:
        resp = TestClient(app).post("/qa", json={"question": "alpha"})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 503
    assert "does not match the embedding model" in resp.json()["detail"]


def test_stats_route_shows_the_model_and_size_of_the_index(tmp_path: Path) -> None:
    _seed_index(tmp_path)
    store = _store(tmp_path, dim=8, model="model-one")
    app.dependency_overrides[get_vector_store] = lambda: store
    try:
        body = TestClient(app).get("/stats").json()
    finally:
        app.dependency_overrides.clear()
    assert (body["embedding_model"], body["embedding_dimension"]) == ("model-one", 8)


# --- pinning the Gemini model ------------------------------------------------------


class _FakeGenai:
    def __init__(self, failing: set | None = None) -> None:
        self.failing = failing or set()
        self.models_used: list = []

    def embed_content(self, model: str, content: str) -> dict:
        self.models_used.append(model)
        if model in self.failing:
            raise RuntimeError("503 model unavailable")
        return {"embedding": [0.1, 0.2, 0.3]}


def _gemini(model: str | None, genai: _FakeGenai) -> GeminiEmbeddingBackend:
    backend = GeminiEmbeddingBackend(api_key="k", delay_seconds=0, model=model)
    backend._genai = genai
    return backend


def test_a_pinned_model_is_used_as_is_without_probing_others() -> None:
    genai = _FakeGenai()
    backend = _gemini("gemini-embedding-001", genai)
    backend.encode(["a", "b"])
    assert genai.models_used == ["models/gemini-embedding-001"] * 2
    assert backend.model_name == "models/gemini-embedding-001"


def test_a_pinned_model_that_fails_is_an_error_not_a_silent_switch() -> None:
    genai = _FakeGenai(failing={"models/gemini-embedding-001"})
    backend = _gemini("models/gemini-embedding-001", genai)
    with pytest.raises(RuntimeError, match="model unavailable"):
        backend.encode(["a"])
    assert set(genai.models_used) == {"models/gemini-embedding-001"}


def test_without_a_pinned_model_the_old_fallback_still_works() -> None:
    genai = _FakeGenai(failing={CANDIDATE_MODELS[0]})
    backend = _gemini(None, genai)
    assert backend.model_name is None
    backend.encode(["a"])
    assert backend.model_name == CANDIDATE_MODELS[1]


def test_service_reports_the_model_that_really_made_the_vectors() -> None:
    assert EmbeddingService(backend=_Backend(3)).active_model_name == EmbeddingService().model_name
    assert EmbeddingService(backend=_Backend(3, "special")).active_model_name == "special"
    genai = _FakeGenai()
    svc = EmbeddingService(backend=_gemini("gemini-embedding-001", genai))
    assert svc.active_model_name == "models/gemini-embedding-001"


def test_factory_pins_the_model_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = build_embedding_service(
        SimpleNamespace(gemini_api_key=SimpleNamespace(get_secret_value=lambda: "k"),
                        gemini_embedding_model="gemini-embedding-001"))
    assert pinned.active_model_name == "models/gemini-embedding-001"
    unpinned = build_embedding_service(
        SimpleNamespace(gemini_api_key=SimpleNamespace(get_secret_value=lambda: "k"),
                        gemini_embedding_model="  "))
    assert unpinned.backend.model_name is None
    local = build_embedding_service(
        SimpleNamespace(gemini_api_key=SimpleNamespace(get_secret_value=lambda: ""),
                        gemini_embedding_model="x"))
    assert local._backend is None


def test_the_model_setting_defaults_to_empty_and_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_EMBEDDING_MODEL", raising=False)
    assert Settings(_env_file=None).gemini_embedding_model == ""
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    assert Settings(_env_file=None).gemini_embedding_model == "gemini-embedding-001"
