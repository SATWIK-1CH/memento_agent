"""Retry helpers that distinguish daily quota exhaustion from burst 429s."""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from typing import TypeVar

from memento_agent.errors import LLMError, QuotaExhaustedError, RateLimitError

T = TypeVar("T")

_DAILY_MARKERS = (
    "perday",
    "per day",
    "per_day",
    "rpd",
    "requests per day",
    "generatecontentrequestsdaily",
    "freetierrequestsperday",
    "generate_content_free_tier_requests_per_day",
)

_GROUNDING_UNAVAILABLE = (
    "google search grounding is not supported",
    "google_search is not supported",
    "tool google_search is not available",
    "search grounding",
    "grounding is not available",
)


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 4,
    initial_delay_s: float = 1.5,
    max_delay_s: float = 32.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn``, backing off on transient 429/5xx. Daily quota is not retried."""
    delay = initial_delay_s
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except QuotaExhaustedError:
            raise
        except Exception as exc:
            last_exc = exc
            mapped = classify_llm_error(exc)
            if isinstance(mapped, QuotaExhaustedError):
                raise mapped from exc
            if isinstance(mapped, RateLimitError) and attempt < max_attempts:
                wait = _retry_delay_seconds(exc, fallback=delay)
                wait = min(max(wait, 0.5), max_delay_s) + random.uniform(0, 0.35)
                sleeper(wait)
                delay = min(delay * 2, max_delay_s)
                continue
            raise mapped from exc
    assert last_exc is not None
    raise classify_llm_error(last_exc)


def classify_llm_error(exc: BaseException) -> Exception:
    """Map SDK/http errors onto Memento errors with actionable messages."""
    from memento_agent.errors import MementoError

    if isinstance(exc, MementoError):
        return exc
    code = getattr(exc, "code", None)
    message = _error_blob(exc)
    lower = message.lower()

    if any(marker in lower for marker in _GROUNDING_UNAVAILABLE) or (
        "google_search" in lower and ("not available" in lower or "not supported" in lower)
    ):
        return LLMError(
            "Google Search grounding is not available for this model on the current Gemini "
            "API tier. On the free tier, Grounding with Google Search works with "
            "gemini-2.5-flash and gemini-2.5-flash-lite (about 500 grounded requests/day "
            "shared). Gemini 3+ models require billing for search. Set --model "
            "gemini-2.5-flash or MEMENTO_SEARCH_MODEL=gemini-2.5-flash."
        )

    if code == 429 or "resource_exhausted" in lower or "resource exhausted" in lower:
        if _looks_like_daily_quota(lower):
            return QuotaExhaustedError(
                "Gemini free-tier quota is exhausted for today. Grounding with Google Search "
                "is capped at about 500 requests/day on the free tier (shared across "
                "gemini-2.5-flash and gemini-2.5-flash-lite) and resets at midnight Pacific "
                "Time. Wait until reset, or enable billing in Google AI Studio."
            )
        return RateLimitError(
            "Gemini rate-limited this request (HTTP 429). The agent retried with backoff "
            "and still got 429. Wait a minute and try again; free-tier RPM/TPM limits "
            "are per project. See https://ai.google.dev/gemini-api/docs/rate-limits"
        )

    if code in {401, 403} or "api key not valid" in lower or "permission denied" in lower:
        return LLMError(
            "Gemini rejected the API key. Check GEMINI_API_KEY at "
            "https://aistudio.google.com/apikey"
        )

    if code in {500, 502, 503, 504}:
        return RateLimitError(
            f"Gemini returned a transient server error ({code}). Wait a moment and retry."
        )

    return LLMError(f"Gemini request failed: {message or type(exc).__name__}")


def _looks_like_daily_quota(blob: str) -> bool:
    compact = blob.replace(" ", "").replace("-", "").replace("_", "").lower()
    if any(marker.replace(" ", "") in compact for marker in _DAILY_MARKERS):
        return True
    if "limit: 0" in blob or "quota value: 0" in blob:
        return True
    return False


def _error_blob(exc: BaseException) -> str:
    parts = [str(exc)]
    for attr in ("message", "status", "details"):
        value = getattr(exc, attr, None)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def _retry_delay_seconds(exc: BaseException, *, fallback: float) -> float:
    blob = _error_blob(exc)
    match = re.search(r"retry in ([\d.]+)\s*s", blob, re.I)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return fallback
    match = re.search(r'"retryDelay":\s*"(\d+)s"', blob)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return fallback
    details = getattr(exc, "details", None)
    delay = _retry_delay_from_details(details)
    return delay if delay is not None else fallback


def _retry_delay_from_details(details: object) -> float | None:
    if details is None:
        return None
    items: list[object]
    if isinstance(details, dict):
        inner = details.get("error", details)
        if isinstance(inner, dict) and "details" in inner:
            items = list(inner.get("details") or [])
        else:
            items = [details]
    elif isinstance(details, list):
        items = details
    else:
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("@type", "")).endswith("RetryInfo"):
            raw = item.get("retryDelay") or item.get("retry_delay")
            if isinstance(raw, str) and raw.endswith("s"):
                try:
                    return float(raw[:-1])
                except ValueError:
                    return None
            if isinstance(raw, (int, float)):
                return float(raw)
    return None
