"""Shared plumbing for live data providers: fetching, caching, and provenance.

Design constraints this module exists to satisfy, in order of importance:

* **The app must never depend on an external endpoint being up.** Every fetch is
  wrapped so that a timeout, a 404, a moved endpoint or a changed payload shape
  becomes a :class:`ProviderResult` carrying validation errors — never an
  exception that reaches the UI. A provider that fails is reported as failed and
  the rest of the app carries on with whatever other sources succeeded.

* **No scraping that violates a platform's terms.** Every provider shipped here
  reads a public JSON endpoint that returns data without authentication, or one
  the user has supplied their own credentials for. No provider parses HTML, and
  none circumvents a paywall, a login, or a rate limit.

* **Nothing is presented as fresher than it is.** Every result carries the URL it
  came from and the timestamp it was fetched, and cached reads are labelled with
  the age of the cache rather than the time of the call.

Providers are deliberately dumb: each one fetches and shapes a single source into
a DataFrame. Joining sources together is :mod:`services.providers.resolver`'s
job, not theirs.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import pandas as pd

from core.config import DEFAULT_PATHS
from core.validation import ValidationReport

LOGGER = logging.getLogger("fantasy_mock_draft.providers")

# A descriptive agent rather than a browser string: these are public JSON
# endpoints, and identifying the client honestly is the courteous way to use
# them. Impersonating a browser would be a step toward evading a block, which is
# exactly what this module refuses to do.
USER_AGENT = "fantasy-mock-draft/1.0 (personal draft tool; +local use)"

DEFAULT_TIMEOUT_SECONDS = 25.0
DEFAULT_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5

# Stands in for a payload that has been parsed and released. `FetchOutcome.ok` is
# "payload is not None", so a consumed payload needs a truthy placeholder rather
# than None — see the note in `fetch_json`.
PAYLOAD_CONSUMED = b"<parsed>"

# Twelve hours. ADP moves over days, not minutes, and every one of these
# endpoints is someone else's infrastructure being used for free — re-fetching
# on every rerun would be both slower for the user and rude to the host.
DEFAULT_CACHE_TTL_SECONDS = 12 * 60 * 60


def cache_directory() -> str:
    """Where fetched payloads are cached. Created on first use."""
    path = DEFAULT_PATHS.cache or os.path.join(DEFAULT_PATHS.root, "data", "cache")
    os.makedirs(path, exist_ok=True)
    return path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.replace(microsecond=0).isoformat()


@dataclass(slots=True)
class FetchOutcome:
    """The raw result of one HTTP fetch, successful or not."""

    payload: bytes | None
    url: str
    from_cache: bool = False
    fetched_at: str = ""
    cache_age_seconds: float | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.payload is not None


@dataclass(slots=True)
class ProviderResult:
    """A shaped table from one source, plus everything needed to judge it.

    ``frame`` is empty when the fetch failed; ``report`` then carries the reason.
    Callers are expected to check :attr:`ok` and carry on with other sources
    rather than treating a failure as fatal.
    """

    frame: pd.DataFrame
    source: str
    url: str = ""
    fetched_at: str = ""
    from_cache: bool = False
    cache_age_seconds: float | None = None
    season: int | None = None
    scoring_format: str = ""
    report: ValidationReport = field(default_factory=ValidationReport)
    notes: str = ""

    @property
    def ok(self) -> bool:
        return self.report.ok and not self.frame.empty

    @property
    def row_count(self) -> int:
        return 0 if self.frame is None else int(len(self.frame))

    def freshness_label(self) -> str:
        """A human sentence about how current this data is.

        Written for display next to the data itself, because "ADP" with no age on
        it invites the reader to assume it is live when it may be half a day old.
        """
        if not self.fetched_at:
            return "age unknown"
        if not self.from_cache:
            return f"fetched live at {self.fetched_at}"
        age = self.cache_age_seconds or 0.0
        if age < 90 * 60:
            return f"cached {int(age / 60)} min ago"
        return f"cached {age / 3600:.1f} hours ago"


class DataProvider(Protocol):
    """One external source of player rankings, ADP, or league data."""

    key: str
    label: str
    description: str
    requires_credentials: bool

    def fetch(self, **kwargs: Any) -> ProviderResult: ...


# ─────────────────────────────────────────────────────────────────────────────
# Fetching
# ─────────────────────────────────────────────────────────────────────────────
def _cache_path(cache_key: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in cache_key)[:180]
    return os.path.join(cache_directory(), f"{safe}.json.gz")


def read_cache(
    cache_key: str, *, ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS
) -> tuple[bytes | None, float | None]:
    """Return cached bytes and their age, or ``(None, None)`` if unusable.

    A corrupt or unreadable cache file is treated as a cache miss rather than an
    error: the worst case is one extra fetch, and failing here would break the
    app for the sake of a temporary file.
    """
    path = _cache_path(cache_key)
    try:
        age = time.time() - os.path.getmtime(path)
        if ttl_seconds >= 0 and age > ttl_seconds:
            return None, age
        with gzip.open(path, "rb") as handle:
            return handle.read(), age
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        LOGGER.debug("Cache miss for %s: %s", cache_key, exc)
        return None, None


def write_cache(cache_key: str, payload: bytes) -> None:
    """Persist a payload. Failures are logged and ignored — the cache is a
    convenience, and a read-only disk should not stop the app working."""
    try:
        with gzip.open(_cache_path(cache_key), "wb") as handle:
            handle.write(payload)
    except OSError as exc:
        LOGGER.warning("Could not write cache for %s: %s", cache_key, exc)


def clear_cache() -> int:
    """Delete every cached payload. Returns how many files were removed."""
    removed = 0
    directory = cache_directory()
    for name in os.listdir(directory):
        if not name.endswith(".json.gz"):
            continue
        try:
            os.remove(os.path.join(directory, name))
            removed += 1
        except OSError as exc:
            LOGGER.warning("Could not delete cache file %s: %s", name, exc)
    LOGGER.info("Cleared %d cached payload(s)", removed)
    return removed


def cache_entries() -> list[dict[str, Any]]:
    """Describe what is currently cached, for display on the Setup page."""
    entries: list[dict[str, Any]] = []
    directory = cache_directory()
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json.gz"):
            continue
        path = os.path.join(directory, name)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        entries.append({
            "key": name[: -len(".json.gz")],
            "size_kb": round(stat.st_size / 1024, 1),
            "age_hours": round((time.time() - stat.st_mtime) / 3600, 2),
            "fetched_at": _iso(datetime.fromtimestamp(stat.st_mtime, timezone.utc)),
        })
    return entries


def fetch_bytes(
    url: str,
    *,
    cache_key: str,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    force_refresh: bool = False,
) -> FetchOutcome:
    """GET a URL with caching and retries, converting every failure to a message.

    This function does not raise. A caller that gets ``ok is False`` should
    report :attr:`FetchOutcome.error` and continue with other sources.

    On a failed fetch, an *expired* cache entry is used if one exists: stale data
    that is labelled stale is more useful than nothing, and the label travels
    with it via :attr:`FetchOutcome.cache_age_seconds`.
    """
    if not url.lower().startswith("https://"):
        return FetchOutcome(None, url, error="Only https:// URLs are fetched.")

    if not force_refresh:
        cached, age = read_cache(cache_key, ttl_seconds=ttl_seconds)
        if cached is not None:
            LOGGER.debug("Cache hit for %s (%.0fs old)", cache_key, age or 0)
            return FetchOutcome(
                cached, url, from_cache=True, fetched_at=_iso(_utc_now()),
                cache_age_seconds=age,
            )

    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    request_headers.update(headers or {})

    last_error = ""
    for attempt in range(1, max(1, retries) + 1):
        try:
            request = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    payload = gzip.decompress(payload)
            write_cache(cache_key, payload)
            LOGGER.info(
                "Fetched %s (%.1f KB, attempt %d)", url, len(payload) / 1024, attempt
            )
            return FetchOutcome(payload, url, fetched_at=_iso(_utc_now()))
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code} {exc.reason}"
            # 4xx means the request itself is wrong (endpoint moved, filter
            # rejected, auth needed). Retrying an identical request cannot help
            # and only adds load to someone else's server.
            if 400 <= exc.code < 500 and exc.code != 429:
                break
        except urllib.error.URLError as exc:
            last_error = f"network error: {exc.reason}"
        except TimeoutError:
            last_error = f"timed out after {timeout_seconds:.0f}s"
        except Exception as exc:  # noqa: BLE001 — any failure must stay non-fatal
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < retries:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    LOGGER.warning("Fetch failed for %s: %s", url, last_error)

    # Fall back to stale cache rather than nothing.
    stale, age = read_cache(cache_key, ttl_seconds=-1)
    if stale is not None:
        LOGGER.info("Serving stale cache for %s (%.1f hours old)", cache_key, (age or 0) / 3600)
        return FetchOutcome(
            stale, url, from_cache=True, fetched_at=_iso(_utc_now()),
            cache_age_seconds=age,
            error=f"{last_error} — served stale cached copy instead.",
        )
    return FetchOutcome(None, url, error=last_error)


def fetch_json(
    url: str, *, cache_key: str, **kwargs: Any
) -> tuple[Any | None, FetchOutcome]:
    """:func:`fetch_bytes` plus JSON decoding. Decode failures are non-fatal.

    A payload that stops being JSON is the signature of an endpoint that has
    moved or started returning an HTML error page, which is precisely the
    situation the app has to survive.
    """
    outcome = fetch_bytes(url, cache_key=cache_key, **kwargs)
    if not outcome.ok:
        return None, outcome
    try:
        parsed = json.loads(outcome.payload or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        outcome.error = (
            f"Response was not valid JSON ({exc}). The endpoint may have changed "
            "or returned an error page."
        )
        outcome.payload = None
        return None, outcome
    except MemoryError:
        # ESPN's player endpoint is ~39 MB and needs several times that to decode,
        # so on a machine already under memory pressure this is a real outcome
        # rather than a theoretical one. It has to be caught here: letting it
        # propagate would break the "no provider raises" guarantee the whole
        # module rests on.
        size_mb = len(outcome.payload or b"") / (1024 * 1024)
        outcome.error = (
            f"Ran out of memory decoding a {size_mb:.0f} MB response. Close some "
            "applications and try again, or switch this source off — the other "
            "sources need far less memory."
        )
        outcome.payload = None
        LOGGER.warning("MemoryError decoding %.0f MB from %s", size_mb, url)
        return None, outcome

    # The payload can be tens of megabytes, and holding the bytes *and* the parsed
    # object doubles peak memory while the caller shapes its DataFrame. The bytes
    # are no longer needed, but `ok` is defined as "payload is not None", so the
    # reference is replaced with a marker rather than cleared — clearing it would
    # make every caller's `outcome.ok` check report a successful fetch as failed.
    outcome.payload = PAYLOAD_CONSUMED
    return parsed, outcome


def failed_result(
    source: str, outcome: FetchOutcome, *, hint: str = ""
) -> ProviderResult:
    """Build an empty result that explains itself.

    Used by every provider so a dead source produces one consistent, actionable
    message instead of each provider inventing its own.
    """
    report = ValidationReport()
    message = f"{source} is unavailable: {outcome.error or 'unknown error'}."
    if hint:
        message = f"{message} {hint}"
    report.error(f"{source.lower().replace(' ', '_')}_unavailable", message)
    return ProviderResult(
        frame=pd.DataFrame(), source=source, url=outcome.url,
        fetched_at=outcome.fetched_at, report=report,
    )


__all__ = [
    "USER_AGENT",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_RETRIES",
    "DEFAULT_CACHE_TTL_SECONDS",
    "FetchOutcome",
    "ProviderResult",
    "DataProvider",
    "cache_directory",
    "cache_entries",
    "clear_cache",
    "read_cache",
    "write_cache",
    "fetch_bytes",
    "fetch_json",
    "failed_result",
]
