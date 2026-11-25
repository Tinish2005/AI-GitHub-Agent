"""CORS settings, built from the CORS_ORIGINS and CORS_ORIGIN_REGEX settings.

CORS_ORIGINS
    Comma-separated list of frontend addresses, e.g. ``https://my-app.vercel.app``.
    Empty or ``*`` means "allow every website" (the default).
CORS_ORIGIN_REGEX
    Optional pattern for addresses that change on every deploy, e.g. Vercel
    preview URLs: ``^https://my-app(-[a-z0-9-]+)?\\.vercel\\.app$``.

Browsers refuse the combination "allow every origin" + "allow credentials", so
credentials are switched off whenever every origin is allowed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class CorsConfig:
    allow_origins: list[str]
    allow_origin_regex: str | None
    allow_credentials: bool
    wildcard: bool

    def middleware_kwargs(self) -> dict:
        """Keyword arguments for ``app.add_middleware(CORSMiddleware, ...)``."""
        return {
            "allow_origins": self.allow_origins,
            "allow_origin_regex": self.allow_origin_regex,
            "allow_credentials": self.allow_credentials,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
            # Browsers hide response headers from the page unless they are listed here.
            "expose_headers": ["Retry-After"],
        }


def _clean_origin(raw: str) -> str:
    """Normalise one origin the way browsers send it: lower case, no trailing slash."""
    origin = raw.strip().strip("\"'").strip().rstrip("/").lower()
    if not origin or origin == "*":
        return origin
    parts = urlsplit(origin)
    if parts.scheme not in ("http", "https") or not parts.netloc or parts.path or parts.query or parts.fragment:
        raise ValueError(
            f"CORS_ORIGINS has an invalid entry: {raw.strip()!r}. "
            "Use the full address with no path, like https://my-app.vercel.app"
        )
    return origin


def parse_origins(raw: str | None) -> list[str]:
    """Split a comma-separated list, clean each entry and drop blanks and repeats."""
    origins: list[str] = []
    for part in (raw or "").split(","):
        origin = _clean_origin(part)
        if origin and origin not in origins:
            origins.append(origin)
    return origins


def build_cors_config(origins_raw: str | None, regex_raw: str | None = "") -> CorsConfig:
    """Turn the two settings into middleware options. Raises ValueError on a bad setting."""
    origins = parse_origins(origins_raw)
    regex = (regex_raw or "").strip().strip("\"'") or None
    if regex is not None:
        try:
            re.compile(regex)
        except re.error as exc:
            raise ValueError(f"CORS_ORIGIN_REGEX is not a valid regular expression: {exc}") from exc

    wildcard = "*" in origins or (not origins and regex is None)
    if wildcard:
        return CorsConfig(["*"], None, allow_credentials=False, wildcard=True)
    return CorsConfig(origins, regex, allow_credentials=True, wildcard=False)
