"""Measure how long a sleeping Render server takes to wake up.

Run it against the deployed server after it has been idle for about 15 minutes
(Render's free plan puts a service to sleep after that). Before that, the server is
already awake and the number means nothing; the script says so when it can tell.

    python scripts\\check_cold_start.py https://your-service.onrender.com
    python scripts\\check_cold_start.py https://your-service.onrender.com --origin https://your-app.vercel.app

It polls GET /health until the answer says status "ok", prints how long that took,
then times two normal requests (the first loads the index, the second is cached).
Only the Python standard library is needed.

Exit codes: 0 all good, 1 the server did not wake in time, 2 wrong address or bad usage,
3 awake but a normal request or the CORS check failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ALREADY_AWAKE_SECONDS = 60  # a server up longer than this was not asleep when we started


def fetch(url: str, timeout: float, headers: dict | None = None) -> tuple:
    """GET a URL. Returns (status, body_text, response_headers, seconds). status 0 = network error."""
    started = time.monotonic()
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body, dict(response.headers), time.monotonic() - started
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body, dict(exc.headers), time.monotonic() - started
    except Exception as exc:  # timeout, refused, DNS, TLS: all mean "not awake yet"
        return 0, type(exc).__name__, {}, time.monotonic() - started


def parse_health(status: int, body: str) -> dict | None:
    """The health payload when the server really says ok, else None."""
    if status != 200:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None  # a 200 from something else (a proxy or holding page)
    return data if isinstance(data, dict) and data.get("status") == "ok" else None


def wait_until_ready(base: str, max_wait: float, poll_interval: float, poll_timeout: float, say=print) -> tuple:
    """Poll /health. Returns (health_or_None, seconds, attempts)."""
    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        status, body, _headers, took = fetch(f"{base}/health", poll_timeout)
        elapsed = time.monotonic() - started
        health = parse_health(status, body)
        if health is not None:
            say(f"  attempt {attempt:>2}  {elapsed:6.1f}s  ok")
            return health, elapsed, attempt
        if 400 <= status < 500:
            say(f"  attempt {attempt:>2}  {elapsed:6.1f}s  HTTP {status}: is this the right address? (/health was not found)")
            return None, elapsed, -attempt  # negative: wrong address, do not keep waiting
        what = f"HTTP {status}" if status else f"no answer ({body})"
        say(f"  attempt {attempt:>2}  {elapsed:6.1f}s  waking up: {what}")
        if elapsed >= max_wait:
            return None, elapsed, attempt
        time.sleep(max(0.0, min(poll_interval - took, max_wait - elapsed)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time a Render cold start.")
    parser.add_argument("url", help="Server address, e.g. https://your-service.onrender.com")
    parser.add_argument("--max-wait", type=float, default=120, help="Give up after this many seconds (default 120).")
    parser.add_argument("--poll-interval", type=float, default=3, help="Seconds between attempts (default 3).")
    parser.add_argument("--poll-timeout", type=float, default=5, help="Timeout of each attempt (default 5).")
    parser.add_argument("--path", default="/stats", help="Normal request to time afterwards (default /stats).")
    parser.add_argument("--origin", default="", help="Frontend address; also checks the server allows it (CORS).")
    args = parser.parse_args(argv)

    base = args.url.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        print("The address must start with http:// or https://")
        return 2

    print("Cold start check for", base)
    print("Only meaningful if the server has been idle for about 15 minutes.\n")
    health, seconds, attempts = wait_until_ready(base, args.max_wait, args.poll_interval, args.poll_timeout)
    if health is None:
        if attempts < 0:
            return 2
        print(f"\nFAILED: no 'ok' from /health after {seconds:.0f} s ({attempts} attempts).")
        print("If the server is deployed and running, check CORS and the address too.")
        return 1

    print(f"\nWake-up time: {seconds:.1f} s ({attempts} attempt{'s' if attempts != 1 else ''})")
    print(
        f"Server reports: version {health.get('version')}, up {health.get('uptime_seconds')} s, "
        f"demo restore '{health.get('demo_restore')}' ({health.get('restore_seconds')} s)"
    )
    uptime = health.get("uptime_seconds")
    if isinstance(uptime, (int, float)) and uptime > ALREADY_AWAKE_SECONDS:
        print(f"WARNING: the server had already been up for {uptime:.0f} s, so this was NOT a cold start.")

    code = 0
    print(f"\nNormal request: GET {args.path}")
    for label in ("first (loads the index)", "second (cached)"):
        status, _body, _headers, took = fetch(base + args.path, 30)
        print(f"  {label:<24} HTTP {status}  {took * 1000:7.0f} ms")
        if not 200 <= status < 300:
            code = 3

    if args.origin:
        origin = args.origin.strip().rstrip("/")
        status, _body, headers, _took = fetch(f"{base}/health", 15, {"Origin": origin})
        allowed = {k.lower(): v for k, v in headers.items()}.get("access-control-allow-origin", "")
        ok = allowed in ("*", origin)
        print(f"\nCORS for {origin}: {'allowed' if ok else 'BLOCKED'}" + ("" if ok else f" (server sent {allowed or 'nothing'}). Check CORS_ORIGINS."))
        if not ok:
            code = 3
    return code


if __name__ == "__main__":
    sys.exit(main())
