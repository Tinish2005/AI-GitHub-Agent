"""Loop 8 tests: visitor indexing jobs, their limits, and the public routes (OFFLINE)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from backend.cloning.cloner import FakeCloner
from backend.cloning.indexing_service import IndexingService
from backend.config import Settings, get_settings
from backend.github.repo_check import RepoCheckError
from backend.github.repo_indexer import GitHubRepoIndexer
from backend.indexing.ast_parser import parse_source
from backend.indexing.jobs import START_MESSAGE, IndexJobManager
from backend.indexing.vector_store import EmbeddingMismatchError
from backend.limits import LimitError, LimitGuard
from backend.main import app, get_indexing_service, get_vector_store
from tests.test_index_safety import ALPHA, BETA, _store
from tests.test_limits import Clock

FILES = {"src/a.py": ALPHA, "src/b.py": BETA}
KEY = "admin-key-for-tests"


def settings(**kw) -> Settings:
    kw.setdefault("public_indexing_enabled", True)
    return Settings(_env_file=None, **kw)


def checker(owner: str, repo: str) -> dict:
    return {"full_name": f"{owner}/{repo}", "default_branch": "main"}


def build(tmp_path: Path, files: dict = FILES, *, run_in_thread: bool = False, check=checker,
          guard_clock=None, hook=None, clock=None, **settings_kw):
    cfg = settings(**settings_kw)
    guard = LimitGuard(cfg, clock=guard_clock or time.monotonic) if guard_clock else LimitGuard(cfg)
    store_dir = tmp_path / "vdb"

    def store_factory():
        return _store(store_dir)

    def indexer_factory(store, max_files):
        ix = GitHubRepoIndexer(token=None, vector_store=store, max_files=max_files)
        ix._default_branch = lambda o, r: "main"
        ix._list_python_files = lambda o, r, ref: sorted(files)

        def fetch(o, r, path, ref):
            if hook:
                hook(path)
            return files[path]

        ix._fetch_file = fetch
        return ix

    manager = IndexJobManager(
        guard=guard, settings=cfg, store_factory=store_factory, indexer_factory=indexer_factory,
        repo_checker=check, run_in_thread=run_in_thread, **({"clock": clock} if clock else {}),
    )
    return SimpleNamespace(manager=manager, guard=guard, settings=cfg, store=store_factory, dir=store_dir)


def refusal(env, owner="o", repo="r", ip="1.1.1.1", admin=False) -> LimitError:
    with pytest.raises(LimitError) as exc:
        env.manager.submit(owner, repo, ip, admin)
    return exc.value


def add_labelled(store, count: int, repo: str, at: str, demo: bool = False) -> None:
    src = "".join(f"def f{repo.replace('/', '_')}_{i}():\n    return {i}\n\n" for i in range(count))
    tags = {"repo": repo, "indexed_at": at}
    if demo:
        tags["demo"] = "true"
    store.add_chunks(parse_source(src, Path(f"{repo.replace('/', '_')}.py")), extra_metadata=tags)


# --- a normal job ------------------------------------------------------------------------------


def test_a_job_indexes_the_repo_and_reports_what_it_did(tmp_path: Path) -> None:
    env = build(tmp_path)
    view = env.manager.submit("o", "r", "1.1.1.1")
    assert view["status"] == "done" and view["phase"] == "finished" and view["repo"] == "o/r"
    assert (view["files_indexed"], view["chunks_indexed"], view["chunks_total"]) == (2, 2, 2)
    assert view["estimated_seconds"] == int(2 * 0.8) and view["error"] == ""
    assert view["message"].startswith("Indexed 2 files into 2 chunks")
    labels = {m["repo"] for m in env.store().all_metadata()}
    assert labels == {"o/r"}
    assert not env.guard.index_lock.locked()
    assert env.manager.get(view["job_id"])["status"] == "done"


def test_the_start_message_says_it_takes_a_few_minutes() -> None:
    assert "few minutes" in START_MESSAGE and "job id" in START_MESSAGE


def test_an_unknown_job_is_none(tmp_path: Path) -> None:
    assert build(tmp_path).manager.get("nope") is None


# --- who may start one -----------------------------------------------------------------------------


def test_public_indexing_is_closed_unless_it_is_switched_on(tmp_path: Path) -> None:
    env = build(tmp_path, public_indexing_enabled=False)
    err = refusal(env)
    assert err.status == 403 and "closed" in err.message
    assert env.store().count() == 0 and not env.guard.index_lock.locked()


def test_the_admin_can_index_even_when_public_indexing_is_off(tmp_path: Path) -> None:
    env = build(tmp_path, public_indexing_enabled=False)
    assert env.manager.submit("o", "r", "1.1.1.1", is_admin=True)["status"] == "done"


@pytest.mark.parametrize("status", [404, 422, 503])
def test_repo_check_failures_are_passed_on_and_start_nothing(tmp_path: Path, status: int) -> None:
    def failing(owner, repo):
        raise RepoCheckError("nope", status)

    env = build(tmp_path, check=failing)
    err = refusal(env)
    assert (err.status, err.message) == (status, "nope")
    assert not env.guard.index_lock.locked() and env.guard.index_hour.remaining("1.1.1.1") == 1


def test_too_many_attempts_are_refused_even_for_bad_repos(tmp_path: Path) -> None:
    def failing(owner, repo):
        raise RepoCheckError("nope", 404)

    env = build(tmp_path, check=failing, public_index_attempts_per_ip_per_hour=2)
    refusal(env); refusal(env)
    err = refusal(env)
    assert err.status == 429 and "Too many indexing requests" in err.message


# --- the demo and repos that are already there ----------------------------------------------------


def test_the_built_in_demo_cannot_be_indexed_again(tmp_path: Path) -> None:
    env = build(tmp_path)
    add_labelled(env.store(), 3, "psf/requests", "2026-01-01T00:00:00+00:00", demo=True)
    before = (env.dir / "idx" / "store.json").read_bytes()
    err = refusal(env, "psf", "requests")
    assert err.status == 409 and "demo" in err.message
    assert (env.dir / "idx" / "store.json").read_bytes() == before


def test_an_already_indexed_repo_answers_at_once_and_costs_nothing(tmp_path: Path) -> None:
    env = build(tmp_path)
    add_labelled(env.store(), 4, "o/r", "2026-01-01T00:00:00+00:00")
    view = env.manager.submit("O", "R", "1.1.1.1")           # the capitals do not matter
    assert view["status"] == "already_indexed" and view["job_id"] is None and view["chunks_indexed"] == 4
    assert env.guard.index_hour.remaining("1.1.1.1") == 1 and env.guard.embed_day.remaining() == 400
    assert not env.guard.index_lock.locked()


# --- one at a time, and per-address limits --------------------------------------------------------------


def test_while_one_job_runs_another_is_told_to_wait_and_is_not_charged(tmp_path: Path) -> None:
    env = build(tmp_path)
    env.guard.index_lock.acquire()
    err = refusal(env, "o", "other")
    assert err.status == 429 and err.retry_after == 60 and "Another repository" in err.message
    assert env.guard.index_hour.remaining("1.1.1.1") == 1
    env.guard.index_lock.release()


def test_one_address_is_limited_per_hour_and_per_day_but_others_are_not(tmp_path: Path) -> None:
    clock = Clock()
    env = build(tmp_path, guard_clock=clock, public_index_per_ip_per_hour=1, public_index_per_ip_per_day=2)
    assert env.manager.submit("o", "one", "1.1.1.1")["status"] == "done"
    err = refusal(env, "o", "two")
    assert err.status == 429 and err.retry_after > 0
    assert env.manager.submit("o", "two", "2.2.2.2")["status"] == "done"      # another address
    clock.advance(3601)
    assert env.manager.submit("o", "three", "1.1.1.1")["status"] == "done"
    clock.advance(3601)
    assert refusal(env, "o", "four").status == 429                           # the daily limit
    assert not env.guard.index_lock.locked()


def test_the_admin_is_not_limited_per_address(tmp_path: Path) -> None:
    env = build(tmp_path)
    for name in ("a", "b", "c"):
        assert env.manager.submit("o", name, "1.1.1.1", is_admin=True)["status"] == "done"


# --- the daily embedding budget and the size of the store ------------------------------------------------------


def test_when_the_daily_budget_is_used_up_nothing_is_embedded_and_the_job_says_why(tmp_path: Path) -> None:
    env = build(tmp_path, public_index_embeddings_per_day=1)
    view = env.manager.submit("o", "r", "1.1.1.1")
    assert view["status"] == "failed" and "daily indexing budget" in view["error"]
    assert env.store().count() == 0 and env.guard.embed_day.remaining() == 1
    assert not env.guard.index_lock.locked()


def test_the_budget_is_charged_for_the_chunks_that_were_indexed(tmp_path: Path) -> None:
    env = build(tmp_path, public_index_embeddings_per_day=10)
    env.manager.submit("o", "r", "1.1.1.1")
    assert env.guard.embed_day.remaining() == 8


def test_old_visitor_repos_are_removed_to_make_room_and_the_demo_never_is(tmp_path: Path) -> None:
    env = build(tmp_path, public_index_store_max_chunks=8)
    store = env.store()
    add_labelled(store, 3, "psf/requests", "2026-01-01T00:00:00+00:00", demo=True)
    add_labelled(store, 2, "old/repo", "2026-02-01T00:00:00+00:00")
    add_labelled(store, 2, "newer/repo", "2026-03-01T00:00:00+00:00")
    view = env.manager.submit("o", "r", "1.1.1.1")                 # needs 2 more chunks; 7 + 2 > 8
    assert view["status"] == "done"
    repos = set(env.store().repo_summaries())
    assert repos == {"psf/requests", "newer/repo", "o/r"}          # the oldest visitor repo went
    assert env.store().count() <= 8


def test_if_even_removing_every_visitor_repo_is_not_enough_the_job_fails_and_the_demo_stays(tmp_path: Path) -> None:
    env = build(tmp_path, public_index_store_max_chunks=4)
    store = env.store()
    add_labelled(store, 3, "psf/requests", "2026-01-01T00:00:00+00:00", demo=True)
    add_labelled(store, 1, "old/repo", "2026-02-01T00:00:00+00:00")
    view = env.manager.submit("o", "r", "1.1.1.1")
    assert view["status"] == "failed" and "full" in view["error"]
    assert "psf/requests" in env.store().repo_summaries() and env.store().repo_summaries()["psf/requests"]["chunks"] == 3
    assert env.guard.embed_day.remaining() == 400                    # nothing was charged


# --- limits on what is indexed ----------------------------------------------------------------------------------


def test_the_chunk_limit_and_skipped_files_are_reported(tmp_path: Path) -> None:
    env = build(tmp_path, public_index_max_chunks=1)
    view = env.manager.submit("o", "r", "1.1.1.1")
    assert view["status"] == "done" and view["chunks_indexed"] == 1
    assert (view["files_skipped"], view["truncated"]) == (1, True) and "skipped" in view["message"]


def test_the_file_limit_comes_from_the_settings(tmp_path: Path) -> None:
    env = build(tmp_path, public_index_max_files=1)
    assert env.manager.submit("o", "r", "1.1.1.1")["files_indexed"] == 1


# --- failures ----------------------------------------------------------------------------------------------------------


def test_an_unexpected_error_fails_the_job_without_leaking_details_and_frees_the_lock(tmp_path: Path) -> None:
    def explode(path):
        raise RuntimeError("secret token ghp_abc in this message")

    env = build(tmp_path, hook=explode)
    view = env.manager.submit("o", "r", "1.1.1.1")
    # a failing fetch only skips that file, so no code is found
    assert view["status"] == "failed" and "ghp_abc" not in str(view)
    assert not env.guard.index_lock.locked()


def test_a_crash_inside_the_indexer_is_reported_by_type_only(tmp_path: Path) -> None:
    env = build(tmp_path)
    original = env.manager._indexer_factory

    def broken(store, max_files):
        ix = original(store, max_files)
        ix.index = lambda *a, **k: (_ for _ in ()).throw(KeyError("secret-value"))
        return ix

    env.manager._indexer_factory = broken
    view = env.manager.submit("o", "r", "1.1.1.1")
    assert view["status"] == "failed" and "KeyError" in view["error"] and "secret-value" not in str(view)
    assert not env.guard.index_lock.locked()


def test_a_model_mismatch_fails_the_job_with_a_friendly_message(tmp_path: Path) -> None:
    env = build(tmp_path)
    original = env.manager._indexer_factory

    def mismatch(store, max_files):
        ix = original(store, max_files)
        ix.index = lambda *a, **k: (_ for _ in ()).throw(EmbeddingMismatchError("different model"))
        return ix

    env.manager._indexer_factory = mismatch
    view = env.manager.submit("o", "r", "1.1.1.1")
    assert view["status"] == "failed" and "search settings changed" in view["message"]


def test_a_repo_without_python_code_fails_clearly(tmp_path: Path) -> None:
    env = build(tmp_path, files={"src/e.py": "x = 1\n"})
    view = env.manager.submit("o", "r", "1.1.1.1")
    assert view["status"] == "failed" and "No Python code" in view["error"]


def test_after_a_failure_the_next_job_can_run(tmp_path: Path) -> None:
    env = build(tmp_path, public_index_per_ip_per_hour=5, public_index_per_ip_per_day=5)
    env.guard.embed_day.spend(399)
    assert env.manager.submit("o", "one", "1.1.1.1")["status"] == "failed"
    env.guard.embed_day._used = 0
    assert env.manager.submit("o", "two", "1.1.1.1")["status"] == "done"


# --- housekeeping --------------------------------------------------------------------------------------------------------


def test_finished_jobs_are_forgotten_after_an_hour_and_the_registry_stays_small(tmp_path: Path) -> None:
    clock = Clock(5000.0)
    env = build(tmp_path, clock=clock, public_index_per_ip_per_hour=50, public_index_per_ip_per_day=50)
    first = env.manager.submit("o", "one", "1.1.1.1")["job_id"]
    clock.advance(3601)
    env.manager.submit("o", "two", "1.1.1.1")
    assert env.manager.get(first) is None

    small = build(tmp_path / "s", public_index_per_ip_per_hour=50, public_index_per_ip_per_day=50)
    small.manager._max_jobs = 3
    for i in range(6):
        small.manager.submit("o", f"r{i}", "1.1.1.1")
    assert len(small.manager._jobs) <= 3


# --- threads ---------------------------------------------------------------------------------------------------------------


def test_a_background_job_can_be_watched_while_it_runs(tmp_path: Path) -> None:
    release = threading.Event()
    env = build(tmp_path, run_in_thread=True, hook=lambda path: release.wait(5))
    view = env.manager.submit("o", "r", "1.1.1.1")
    assert view["status"] in ("queued", "running") and "few minutes" in view["message"]
    time.sleep(0.2)
    mid = env.manager.get(view["job_id"])
    assert mid["status"] == "running" and mid["phase"] == "fetching"
    release.set()
    for _ in range(100):
        done = env.manager.get(view["job_id"])
        if done["status"] != "running":
            break
        time.sleep(0.05)
    assert done["status"] == "done" and done["chunks_indexed"] == 2
    assert not env.guard.index_lock.locked()


def test_simultaneous_requests_start_exactly_one_job(tmp_path: Path) -> None:
    release = threading.Event()
    env = build(tmp_path, run_in_thread=True, hook=lambda path: release.wait(5),
                public_index_attempts_per_ip_per_hour=100)
    outcomes: list = []
    barrier = threading.Barrier(8)

    def request(n: int) -> None:
        barrier.wait()
        try:
            outcomes.append(env.manager.submit("o", f"r{n}", f"9.9.9.{n}")["status"])
        except LimitError as exc:
            outcomes.append(exc.status)

    threads = [threading.Thread(target=request, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    release.set()
    started = [o for o in outcomes if isinstance(o, str)]
    assert len(started) == 1 and outcomes.count(429) == 7
    time.sleep(0.5)


# --- the routes ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def api(tmp_path: Path) -> Iterator:
    env = build(tmp_path)
    saved = (app.state.guard, app.state.jobs)
    app.state.guard, app.state.jobs = env.guard, env.manager
    app.dependency_overrides[get_settings] = lambda: settings(admin_api_key=SecretStr(KEY))
    app.dependency_overrides[get_vector_store] = env.store
    env.client = TestClient(app)
    yield env
    app.state.guard, app.state.jobs = saved
    app.dependency_overrides.clear()


def test_the_start_route_answers_202_and_the_status_route_shows_the_job(api) -> None:
    resp = api.client.post("/index/public", json={"owner": "o", "repo": "r"})
    assert resp.status_code == 202
    job = resp.json()
    assert job["status"] == "done" and "job_id" in job and job["repo"] == "o/r"
    status = api.client.get(f"/index/public/{job['job_id']}")
    assert status.status_code == 200 and status.json() == job


def test_the_status_route_404s_for_an_unknown_job(api) -> None:
    assert api.client.get("/index/public/doesnotexist").status_code == 404


@pytest.mark.parametrize("body", [
    {"owner": "../x", "repo": "r"}, {"owner": "o", "repo": "r r"}, {"owner": "", "repo": "r"},
    {"owner": "o"}, {}, {"owner": "o", "repo": "x" * 101},
])
def test_bad_requests_are_refused(api, body: dict) -> None:
    assert api.client.post("/index/public", json=body).status_code == 422


def test_a_closed_server_says_so_but_the_admin_key_still_works(api) -> None:
    api.manager.settings = settings(public_indexing_enabled=False)
    closed = api.client.post("/index/public", json={"owner": "o", "repo": "r"})
    assert closed.status_code == 403 and "closed" in closed.json()["detail"]
    opened = api.client.post("/index/public", json={"owner": "o", "repo": "r"}, headers={"X-Admin-Key": KEY})
    assert opened.status_code == 202


def test_a_limit_gives_429_with_retry_after_in_the_header_and_the_body(api) -> None:
    assert api.client.post("/index/public", json={"owner": "o", "repo": "one"}).status_code == 202
    resp = api.client.post("/index/public", json={"owner": "o", "repo": "two"})
    assert resp.status_code == 429
    assert int(resp.headers["retry-after"]) > 0
    assert resp.json()["retry_after_seconds"] == int(resp.headers["retry-after"])
    assert "limit" in resp.json()["detail"]


def test_the_demo_gives_409_and_an_indexed_repo_gives_200(api) -> None:
    add_labelled(api.store(), 3, "psf/requests", "2026-01-01T00:00:00+00:00", demo=True)
    add_labelled(api.store(), 2, "o/known", "2026-01-01T00:00:00+00:00")
    assert api.client.post("/index/public", json={"owner": "psf", "repo": "requests"}).status_code == 409
    known = api.client.post("/index/public", json={"owner": "o", "repo": "known"})
    assert known.status_code == 200 and known.json()["status"] == "already_indexed"


def test_the_stats_page_shows_the_limits_and_what_is_left(api) -> None:
    api.guard.charge_ai_call("x")
    limits = api.client.get("/stats").json()["limits"]
    assert limits["public_indexing"]["enabled"] is True and limits["public_indexing"]["max_files"] == 10
    assert limits["ai"]["calls_left_today"] == 15 and limits["embeddings"]["index_budget_left_today"] == 400


def test_whoami_needs_the_admin_key_and_shows_how_the_address_was_worked_out(api) -> None:
    assert api.client.get("/admin/whoami").status_code == 403
    plain = api.client.get("/admin/whoami", headers={"X-Admin-Key": KEY, "X-Forwarded-For": "1.2.3.4"}).json()
    assert plain["trusted_proxy_hops"] == 0 and plain["client_ip_used"] == "testclient"   # header ignored
    assert plain["x_forwarded_for"] == "1.2.3.4"
    app.dependency_overrides[get_settings] = lambda: settings(admin_api_key=SecretStr(KEY), trusted_proxy_hops=1)
    trusted = api.client.get("/admin/whoami", headers={"X-Admin-Key": KEY, "X-Forwarded-For": "6.6.6.6, 198.51.100.7"}).json()
    assert trusted["client_ip_used"] == "198.51.100.7" and trusted["direct_peer"] == "testclient"


def test_admin_indexing_waits_for_a_running_job_and_then_gives_a_clear_conflict(api, tmp_path: Path) -> None:
    api.guard.admin_wait_seconds = 0.05
    api.guard.index_lock.acquire()
    app.dependency_overrides[get_indexing_service] = lambda: IndexingService(
        cloner=FakeCloner(cache_dir=tmp_path / "cache"), vector_store=api.store(),
    )
    headers = {"X-Admin-Key": KEY}
    clone = api.client.post("/index", json={"url": "https://github.com/fake/proj.git"}, headers=headers)
    github = api.client.post("/index/github", json={"owner": "o", "repo": "r"}, headers=headers)
    assert clone.status_code == github.status_code == 409
    assert "Another indexing job" in clone.json()["detail"]
    api.guard.index_lock.release()
    ok = api.client.post("/index", json={"url": "https://github.com/fake/proj.git"}, headers=headers)
    assert ok.status_code == 200 and not api.guard.index_lock.locked()
