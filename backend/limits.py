"""Limits that protect the server and your Gemini quota.

Everything here lives in memory: counters reset when the server restarts or
sleeps, and each server process has its own counters.

What is limited:
    - public indexing: attempts and jobs per IP, one job at a time, a daily
      embedding budget
    - AI calls (answers, fixes, plans): per minute, per day and per IP per hour
    - the client IP is taken from a proxy header ONLY when you say how many
      trusted proxies sit in front of the server (TRUSTED_PROXY_HOPS)
"""

from __future__ import annotations

import ipaddress
import threading
import time
from collections import deque
from contextlib import contextmanager


class LimitError(Exception):
    """A limit or a rule stopped the request. The message is safe to show."""

    def __init__(self, message: str, *, status: int = 429, retry_after: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.retry_after = max(int(retry_after), 0)


# ---------------------------------------------------------------------------
# Who is calling
# ---------------------------------------------------------------------------


def normalize_ip(value: str) -> str:
    """One key per caller. IPv6 callers are grouped by /64, since one person owns a whole /64."""
    text = (value or "").strip()
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return text or "unknown"
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


def client_ip(request, trusted_hops: int = 0) -> str:
    """The caller's address.

    With `trusted_hops` = 0 (the default) the X-Forwarded-For header is ignored,
    because a visitor can write anything in it. With N > 0, the address that the
    Nth proxy from the server's side of the chain added is used: that is the Nth
    entry from the RIGHT, which a visitor cannot forge. If the header is shorter
    than expected or not an IP address, the direct connection address is used.
    """
    peer = request.client.host if getattr(request, "client", None) else "unknown"
    if trusted_hops <= 0:
        return normalize_ip(peer)
    header = request.headers.get("x-forwarded-for", "")
    parts = [p.strip() for p in header.split(",") if p.strip()]
    if len(parts) < trusted_hops:
        return normalize_ip(peer)
    candidate = parts[-trusted_hops]
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return normalize_ip(peer)
    return normalize_ip(candidate)


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


class SlidingWindowLimiter:
    """At most `limit` hits per key in any window of `window_seconds`."""

    def __init__(self, limit: int, window_seconds: float, *, clock=time.monotonic, max_keys: int = 10_000) -> None:
        self.limit = int(limit)
        self.window = float(window_seconds)
        self._clock = clock
        self._max_keys = max_keys
        self._hits: dict = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque:
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        return hits

    def peek(self, key: str) -> tuple:
        """(allowed, seconds_until_allowed) without recording a hit."""
        with self._lock:
            now = self._clock()
            hits = self._prune(key, now)
            if len(hits) < self.limit:
                return True, 0
            return False, max(int(hits[0] + self.window - now) + 1, 1)

    def hit(self, key: str) -> tuple:
        """Record a hit if allowed. Returns (allowed, seconds_until_allowed)."""
        with self._lock:
            now = self._clock()
            hits = self._prune(key, now)
            if len(hits) >= self.limit:
                return False, max(int(hits[0] + self.window - now) + 1, 1)
            hits.append(now)
            if len(self._hits) > self._max_keys:
                self._forget_oldest()
            return True, 0

    def remaining(self, key: str) -> int:
        with self._lock:
            return max(self.limit - len(self._prune(key, self._clock())), 0)

    def _forget_oldest(self) -> None:
        # Drop the keys whose newest hit is oldest, so memory stays bounded.
        victims = sorted(self._hits, key=lambda k: self._hits[k][-1] if self._hits[k] else 0)
        for key in victims[: max(len(victims) // 10, 1)]:
            self._hits.pop(key, None)


class DailyBudget:
    """A number that is spent during a UTC day and refilled at midnight UTC."""

    def __init__(self, limit: int, *, clock=time.time) -> None:
        self.limit = int(limit)
        self._clock = clock
        self._day = self._today()
        self._used = 0
        self._lock = threading.Lock()

    def _today(self) -> int:
        return int(self._clock() // 86400)

    def _roll(self) -> None:
        today = self._today()
        if today != self._day:
            self._day, self._used = today, 0

    def peek(self, amount: int = 1) -> bool:
        with self._lock:
            self._roll()
            return self._used + amount <= self.limit

    def spend(self, amount: int = 1) -> bool:
        with self._lock:
            self._roll()
            if self._used + amount > self.limit:
                return False
            self._used += amount
            return True

    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return max(self.limit - self._used, 0)

    def seconds_until_reset(self) -> int:
        return int(86400 - (self._clock() % 86400)) + 1


# ---------------------------------------------------------------------------
# All the limits together
# ---------------------------------------------------------------------------


class LimitGuard:
    """Holds every counter and the one-at-a-time indexing lock."""

    def __init__(self, settings, *, clock=time.monotonic, wall_clock=time.time) -> None:
        self.settings = settings
        self.attempts = SlidingWindowLimiter(settings.public_index_attempts_per_ip_per_hour, 3600, clock=clock)
        self.index_hour = SlidingWindowLimiter(settings.public_index_per_ip_per_hour, 3600, clock=clock)
        self.index_day = SlidingWindowLimiter(settings.public_index_per_ip_per_day, 86400, clock=clock)
        self.ai_minute = SlidingWindowLimiter(settings.ai_calls_per_minute, 60, clock=clock)
        self.ai_ip_hour = SlidingWindowLimiter(settings.ai_calls_per_ip_per_hour, 3600, clock=clock)
        self.ai_day = DailyBudget(settings.ai_calls_per_day, clock=wall_clock)
        self.embed_day = DailyBudget(settings.public_index_embeddings_per_day, clock=wall_clock)
        self.index_lock = threading.Lock()
        self.admin_wait_seconds = 600.0
        self._ai_lock = threading.Lock()

    # --- public indexing -----------------------------------------------------
    def check_attempt(self, ip: str) -> None:
        allowed, wait = self.attempts.hit(ip)
        if not allowed:
            raise LimitError(
                "Too many indexing requests from your address. Try again later.", retry_after=wait,
            )

    def charge_index_job(self, ip: str) -> None:
        """Count one indexing job against the caller's hourly and daily limits."""
        ok_h, wait_h = self.index_hour.peek(ip)
        ok_d, wait_d = self.index_day.peek(ip)
        if not (ok_h and ok_d):
            raise LimitError(
                "You have used your indexing limit for now. Try again later.",
                retry_after=max(wait_h, wait_d),
            )
        self.index_hour.hit(ip)
        self.index_day.hit(ip)

    @contextmanager
    def admin_slot(self, wait_seconds: float | None = None):
        """Admin indexing waits for the lock instead of failing at once."""
        wait = self.admin_wait_seconds if wait_seconds is None else wait_seconds
        if not self.index_lock.acquire(timeout=wait):
            raise LimitError("Another indexing job is running. Try again in a few minutes.", status=409, retry_after=60)
        try:
            yield
        finally:
            self.index_lock.release()

    def check_embeddings(self, chunks: int) -> None:
        """Raise if the daily indexing budget cannot cover `chunks` (spends nothing)."""
        if not self.embed_day.peek(chunks):
            raise LimitError(
                "The daily indexing budget for this demo is used up. Try again tomorrow.",
                retry_after=self.embed_day.seconds_until_reset(),
            )

    def charge_embeddings(self, chunks: int) -> None:
        if not self.embed_day.spend(chunks):
            raise LimitError(
                "The daily indexing budget for this demo is used up. Try again tomorrow.",
                retry_after=self.embed_day.seconds_until_reset(),
            )

    # --- AI calls --------------------------------------------------------------
    def charge_ai_call(self, ip: str, *, is_admin: bool = False) -> None:
        """Count one AI call, or raise LimitError. Nothing is counted if any limit says no."""
        if is_admin:
            return
        with self._ai_lock:
            ok_m, wait_m = self.ai_minute.peek("all")
            ok_i, wait_i = self.ai_ip_hour.peek(ip)
            ok_d = self.ai_day.peek()
            if not ok_d:
                raise LimitError(
                    "The demo's daily AI limit has been reached. It resets at midnight UTC.",
                    retry_after=self.ai_day.seconds_until_reset(),
                )
            if not ok_m:
                raise LimitError("The AI is busy. Try again in a minute.", retry_after=wait_m)
            if not ok_i:
                raise LimitError(
                    "You have used your AI limit for this hour. Try again later.", retry_after=wait_i,
                )
            self.ai_minute.hit("all")
            self.ai_ip_hour.hit(ip)
            self.ai_day.spend()

    # --- for /stats ------------------------------------------------------------
    def snapshot(self) -> dict:
        s = self.settings
        return {
            "public_indexing": {
                "enabled": bool(s.public_indexing_enabled),
                "max_files": s.public_index_max_files,
                "max_chunks": s.public_index_max_chunks,
                "jobs_per_ip_per_hour": s.public_index_per_ip_per_hour,
                "jobs_per_ip_per_day": s.public_index_per_ip_per_day,
                "message": "Indexing takes a few minutes. Only public GitHub repositories can be indexed.",
            },
            "ai": {
                "calls_per_day": s.ai_calls_per_day,
                "calls_left_today": self.ai_day.remaining(),
                "calls_per_minute": s.ai_calls_per_minute,
            },
            "embeddings": {"index_budget_left_today": self.embed_day.remaining()},
        }


# ---------------------------------------------------------------------------
# AI client that counts its calls
# ---------------------------------------------------------------------------


def looks_like_quota_error(exc: Exception) -> bool:
    """True when an AI provider error means 'you are out of quota or going too fast'."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "resourceexhausted" in name or "ratelimit" in name or "toomanyrequests" in name
        or "429" in text or "quota" in text or "rate limit" in text or "rate_limit" in text
    )


class BudgetedLLM:
    """Wraps an AI client: every call is counted, and provider quota errors become clear 429s."""

    def __init__(self, inner, guard: LimitGuard, ip: str, is_admin: bool) -> None:
        self._inner = inner
        self._guard = guard
        self._ip = ip
        self._is_admin = is_admin

    @property
    def model(self) -> str:
        return getattr(self._inner, "model", "unknown")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self._guard.charge_ai_call(self._ip, is_admin=self._is_admin)
        try:
            return self._inner.complete(system_prompt, user_prompt)
        except LimitError:
            raise
        except Exception as exc:
            if looks_like_quota_error(exc):
                raise LimitError(
                    "The AI provider's limit was reached. Try again in a minute, or tomorrow if the daily limit is used up.",
                    retry_after=60,
                ) from exc
            raise
