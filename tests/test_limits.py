"""Loop 8 tests: counters, client IP, the AI-call guard (OFFLINE, with a fake clock)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from backend.config import Settings
from backend.limits import (
    BudgetedLLM,
    DailyBudget,
    LimitError,
    LimitGuard,
    SlidingWindowLimiter,
    client_ip,
    looks_like_quota_error,
    normalize_ip,
)


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


# --- sliding window ---------------------------------------------------------------


def test_the_limit_is_enforced_and_the_window_rolls() -> None:
    clock = Clock()
    limiter = SlidingWindowLimiter(2, 60, clock=clock)
    assert limiter.hit("a") == (True, 0)
    clock.advance(10)
    assert limiter.hit("a") == (True, 0)
    allowed, wait = limiter.hit("a")
    assert allowed is False and wait == 51        # the first hit leaves the window in 50 s
    clock.advance(51)
    assert limiter.hit("a")[0] is True


def test_keys_are_independent_and_peek_records_nothing() -> None:
    limiter = SlidingWindowLimiter(1, 60, clock=Clock())
    assert limiter.peek("a") == (True, 0) and limiter.peek("a") == (True, 0)
    limiter.hit("a")
    assert limiter.peek("a")[0] is False and limiter.hit("b")[0] is True
    assert (limiter.remaining("a"), limiter.remaining("b"), limiter.remaining("new")) == (0, 0, 1)


def test_the_number_of_remembered_keys_stays_bounded() -> None:
    limiter = SlidingWindowLimiter(1, 3600, clock=Clock(), max_keys=50)
    for i in range(500):
        limiter.hit(f"ip-{i}")
    assert len(limiter._hits) <= 60
    assert limiter.hit("fresh")[0] is True


def test_many_threads_cannot_exceed_the_limit() -> None:
    limiter = SlidingWindowLimiter(5, 60, clock=Clock())
    results: list = []

    def work() -> None:
        results.append(limiter.hit("k")[0])

    threads = [threading.Thread(target=work) for _ in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count(True) == 5


# --- daily budget -------------------------------------------------------------------


def test_the_daily_budget_is_spent_and_refills_at_midnight_utc() -> None:
    clock = Clock(86400 * 10 + 100)
    budget = DailyBudget(10, clock=clock)
    assert budget.spend(7) is True and budget.remaining() == 3
    assert budget.peek(3) is True and budget.peek(4) is False
    assert budget.spend(4) is False and budget.remaining() == 3   # a refused spend takes nothing
    assert budget.seconds_until_reset() == 86400 - 100 + 1
    clock.advance(86400)
    assert budget.remaining() == 10 and budget.spend(10) is True


# --- who is calling --------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    ("203.0.113.9", "203.0.113.9"),
    (" 203.0.113.9 ", "203.0.113.9"),
    ("2001:db8:1:2:3:4:5:6", "2001:db8:1:2::/64"),
    ("2001:db8:1:2:ffff::1", "2001:db8:1:2::/64"),
    ("::ffff:203.0.113.9", "203.0.113.9"),
    ("testclient", "testclient"),
    ("", "unknown"),
])
def test_addresses_are_normalized_so_one_person_has_one_key(value: str, expected: str) -> None:
    assert normalize_ip(value) == expected


def req(peer: str | None, forwarded: str | None = None):
    headers = {} if forwarded is None else {"x-forwarded-for": forwarded}
    return SimpleNamespace(client=SimpleNamespace(host=peer) if peer else None, headers=headers)


def test_the_forwarded_header_is_ignored_unless_proxies_are_trusted() -> None:
    forged = req("10.0.0.5", "1.2.3.4")
    assert client_ip(forged, 0) == "10.0.0.5"


def test_with_one_trusted_proxy_the_rightmost_address_is_used_so_forging_does_not_work() -> None:
    assert client_ip(req("10.0.0.5", "198.51.100.7"), 1) == "198.51.100.7"
    # The visitor sends "1.2.3.4"; the proxy adds the real address on the right.
    assert client_ip(req("10.0.0.5", "1.2.3.4, 198.51.100.7"), 1) == "198.51.100.7"
    assert client_ip(req("10.0.0.5", "9.9.9.9, 1.2.3.4, 198.51.100.7"), 1) == "198.51.100.7"


def test_with_two_trusted_proxies_the_second_from_the_right_is_used() -> None:
    assert client_ip(req("10.0.0.5", "6.6.6.6, 198.51.100.7, 172.16.0.1"), 2) == "198.51.100.7"


@pytest.mark.parametrize("peer,header,hops", [
    ("10.0.0.5", None, 1),                  # no header at all
    ("10.0.0.5", "", 1),
    ("10.0.0.5", "198.51.100.7", 2),        # fewer entries than trusted proxies
    ("10.0.0.5", "not-an-ip", 1),           # not an address
    ("10.0.0.5", "1.2.3.4, <script>", 1),
])
def test_a_missing_short_or_invalid_header_falls_back_to_the_direct_address(peer, header, hops) -> None:
    assert client_ip(req(peer, header), hops) == "10.0.0.5"


def test_ipv6_clients_and_missing_peers() -> None:
    assert client_ip(req("10.0.0.5", "2001:db8::1, 198.51.100.7"), 2) == "2001:db8::/64"
    assert client_ip(req(None), 0) == "unknown"
    assert client_ip(req(None, "198.51.100.7"), 1) == "198.51.100.7"


def test_empty_entries_in_the_header_are_ignored() -> None:
    assert client_ip(req("10.0.0.5", " , 198.51.100.7 ,"), 1) == "198.51.100.7"


# --- the guard ------------------------------------------------------------------------------


def guard(**kw) -> tuple:
    clock, wall = Clock(), Clock(86400 * 20 + 50)
    return LimitGuard(settings(**kw), clock=clock, wall_clock=wall), clock, wall


def test_attempts_are_limited_per_address() -> None:
    g, clock, _ = guard(public_index_attempts_per_ip_per_hour=2)
    g.check_attempt("a"); g.check_attempt("a")
    with pytest.raises(LimitError) as exc:
        g.check_attempt("a")
    assert exc.value.status == 429 and exc.value.retry_after > 0
    g.check_attempt("b")
    clock.advance(3601)
    g.check_attempt("a")


def test_an_index_job_counts_against_the_hourly_and_daily_limits_together() -> None:
    g, clock, _ = guard(public_index_per_ip_per_hour=1, public_index_per_ip_per_day=2)
    g.charge_index_job("a")
    with pytest.raises(LimitError):
        g.charge_index_job("a")
    clock.advance(3601)
    g.charge_index_job("a")                       # second of the day
    clock.advance(3601)
    with pytest.raises(LimitError):               # the daily limit is reached
        g.charge_index_job("a")
    clock.advance(86400)
    g.charge_index_job("a")


def test_a_refused_job_charge_counts_nothing() -> None:
    g, clock, _ = guard(public_index_per_ip_per_hour=1, public_index_per_ip_per_day=1)
    g.charge_index_job("a")
    for _ in range(3):
        with pytest.raises(LimitError):
            g.charge_index_job("a")
    assert g.index_hour.remaining("a") == 0 and g.index_day.remaining("a") == 0


def test_the_daily_embedding_budget() -> None:
    g, _, wall = guard(public_index_embeddings_per_day=100)
    g.check_embeddings(100)
    g.charge_embeddings(60)
    g.check_embeddings(40)
    with pytest.raises(LimitError, match="daily indexing budget"):
        g.check_embeddings(41)
    with pytest.raises(LimitError):
        g.charge_embeddings(41)
    assert g.embed_day.remaining() == 40
    wall.advance(86400)
    assert g.embed_day.remaining() == 100


def test_ai_calls_are_limited_per_minute_per_ip_and_per_day() -> None:
    g, clock, _ = guard(ai_calls_per_minute=3, ai_calls_per_ip_per_hour=2, ai_calls_per_day=100)
    g.charge_ai_call("a"); g.charge_ai_call("a")
    with pytest.raises(LimitError, match="this hour"):
        g.charge_ai_call("a")
    g.charge_ai_call("b")
    with pytest.raises(LimitError, match="busy"):
        g.charge_ai_call("c")                       # three calls in the last minute
    clock.advance(61)
    g.charge_ai_call("c")


def test_the_daily_ai_limit_stops_everyone_and_resets_at_midnight() -> None:
    g, _, wall = guard(ai_calls_per_day=2, ai_calls_per_minute=50, ai_calls_per_ip_per_hour=50)
    g.charge_ai_call("a"); g.charge_ai_call("b")
    with pytest.raises(LimitError, match="daily AI limit") as exc:
        g.charge_ai_call("c")
    assert exc.value.retry_after > 0
    wall.advance(86400)
    g.charge_ai_call("c")


def test_a_refused_ai_call_counts_nothing_anywhere() -> None:
    g, _, _ = guard(ai_calls_per_day=10, ai_calls_per_minute=1, ai_calls_per_ip_per_hour=5)
    g.charge_ai_call("a")
    for _ in range(3):
        with pytest.raises(LimitError):
            g.charge_ai_call("b")
    assert g.ai_day.remaining() == 9 and g.ai_ip_hour.remaining("b") == 5


def test_the_admin_is_not_counted() -> None:
    g, _, _ = guard(ai_calls_per_day=1)
    for _ in range(5):
        g.charge_ai_call("a", is_admin=True)
    assert g.ai_day.remaining() == 1


def test_the_admin_lock_waits_then_gives_a_clear_conflict() -> None:
    g, _, _ = guard()
    with g.admin_slot(0.01):
        assert g.index_lock.locked()
        with pytest.raises(LimitError) as exc:
            with g.admin_slot(0.05):
                pass
        assert exc.value.status == 409
    assert not g.index_lock.locked()


def test_the_snapshot_shows_what_is_left() -> None:
    g, _, _ = guard(ai_calls_per_day=16, public_index_embeddings_per_day=400, public_indexing_enabled=True)
    g.charge_ai_call("a")
    g.charge_embeddings(150)
    snap = g.snapshot()
    assert snap["ai"] == {"calls_per_day": 16, "calls_left_today": 15, "calls_per_minute": 4}
    assert snap["embeddings"] == {"index_budget_left_today": 250}
    assert snap["public_indexing"]["enabled"] is True
    assert snap["public_indexing"]["max_files"] == 10 and "few minutes" in snap["public_indexing"]["message"]


def test_the_defaults_match_the_gemini_free_tier_with_room_to_spare() -> None:
    s = settings()
    assert (s.ai_calls_per_day, s.ai_calls_per_minute) == (16, 4)          # the free tier allows 20 and 5
    assert s.public_index_embeddings_per_day == 400                          # the free tier allows 1,000
    assert (s.public_indexing_enabled, s.trusted_proxy_hops) == (False, 0)
    assert (s.public_index_max_files, s.public_index_max_chunks) == (10, 150)


# --- the AI client wrapper --------------------------------------------------------------------


class _LLM:
    model = "stub-model"

    def __init__(self, error: Exception | None = None) -> None:
        self.error, self.calls = error, 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        return "answer"


class ResourceExhausted(Exception):
    pass


@pytest.mark.parametrize("error,expected", [
    (ResourceExhausted("x"), True),
    (RuntimeError("429 Too Many Requests"), True),
    (RuntimeError("You exceeded your current quota"), True),
    (RuntimeError("Rate limit reached for requests"), True),
    (RuntimeError("rate_limit_exceeded"), True),
    (ValueError("bad json"), False),
    (RuntimeError("connection reset"), False),
])
def test_provider_quota_errors_are_recognised(error: Exception, expected: bool) -> None:
    assert looks_like_quota_error(error) is expected


def test_every_ai_call_is_counted_and_passed_through() -> None:
    g, _, _ = guard(ai_calls_per_day=2, ai_calls_per_minute=10, ai_calls_per_ip_per_hour=10)
    inner = _LLM()
    llm = BudgetedLLM(inner, g, "a", False)
    assert llm.model == "stub-model"
    assert llm.complete("s", "u") == "answer" and llm.complete("s", "u") == "answer"
    with pytest.raises(LimitError):
        llm.complete("s", "u")
    assert inner.calls == 2                       # the refused call never reached the provider


def test_a_provider_quota_error_becomes_a_clear_429() -> None:
    g, _, _ = guard()
    llm = BudgetedLLM(_LLM(ResourceExhausted("429 quota exceeded")), g, "a", False)
    with pytest.raises(LimitError) as exc:
        llm.complete("s", "u")
    assert exc.value.status == 429 and exc.value.retry_after == 60 and "AI provider" in exc.value.message


def test_other_provider_errors_are_not_hidden() -> None:
    g, _, _ = guard()
    with pytest.raises(ValueError, match="bad json"):
        BudgetedLLM(_LLM(ValueError("bad json")), g, "a", False).complete("s", "u")


def test_the_admin_can_call_the_ai_without_being_counted() -> None:
    g, _, _ = guard(ai_calls_per_day=1)
    llm = BudgetedLLM(_LLM(), g, "a", True)
    for _ in range(4):
        llm.complete("s", "u")
    assert g.ai_day.remaining() == 1
