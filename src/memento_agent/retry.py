"""Retry helpers that distinguish billing/quota failures from burst 429s."""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from typing import TypeVar

from memento_agent.errors import (
    GroundingUnavailableError,
    LLMError,
    ModelUnavailableError,
    QuotaExhaustedError,
    RateLimitError,
)

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

_BILLING_MARKERS = (
    "plan and billing",
    "check your plan and billing",
    "billing details",
    "enable billing",
)

_GROUNDING_UNAVAILABLE = (
    "google search grounding is not supported",
    "google_search is not supported",
    "tool google_search is not available",
    "search grounding",
    "grounding is not available",
)

_MODEL_GONE = (
    "no longer available to new users",
    "is not found for api version",
    "not found",
)

_FATAL = (QuotaExhaustedError, GroundingUnavailableError, ModelUnavailableError)


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 4,
    initial_delay_s: float = 1.5,
    max_delay_s: float = 32.0,
    sleeper: Callable[[float], None] = time.sleep,
    grounded: bool = False,
) -> T:
    """Call ``fn``, backing off on transient 429/5xx. Billing/quota is not retried."""
    delay = initial_delay_s
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except _FATAL:
            raise
        except Exception as exc:
            last_exc = exc
            mapped = classify_llm_error(exc, grounded=grounded)
            if isinstance(mapped, _FATAL):
                raise mapped from exc
            if isinstance(mapped, RateLimitError) and attempt < max_attempts:
                wait = _retry_delay_seconds(exc, fallback=delay)
                wait = min(max(wait, 0.5), max_delay_s) + random.uniform(0, 0.35)
                sleeper(wait)
                delay = min(delay * 2, max_delay_s)
                continue
            raise mapped from exc
    assert last_exc is not None
    raise classify_llm_error(last_exc, grounded=grounded)


def classify_llm_error(exc: BaseException, *, grounded: bool = False) -> Exception:
    """Map SDK/http errors onto Memento errors with actionable messages."""
    from memento_agent.errors import MementoError

    if isinstance(exc, MementoError):
        return exc
    code = getattr(exc, "code", None)
    message = _error_text(exc)
    lower = message.lower()
    exhausted = code == 429 or "resource_exhausted" in lower or "resource exhausted" in lower

    if _looks_like_model_unavailable(code, lower):
        return ModelUnavailableError(
            "This Gemini model is not available on this API key. New free-tier keys cannot "
            "call gemini-2.5-* (Google's list() still shows them). memento defaults to "
            "gemini-3.5-flash-lite, with gemini-3.8-flash as a fallback. Set MEMENTO_MODEL "
            "to a model that works for your key."
        )

    if _looks_like_grounding_unavailable(lower) or (
        grounded and exhausted and any(marker in lower for marker in _BILLING_MARKERS)
    ):
        return GroundingUnavailableError(_grounding_help())

    if exhausted:
        if _looks_like_daily_quota(lower):
            return QuotaExhaustedError(
                "Gemini free-tier quota is exhausted for today. Limits reset at midnight "
                "Pacific Time. Wait until reset, or try again later. "
                "See https://ai.google.dev/gemini-api/docs/rate-limits"
            )
        return RateLimitError(
            "Gemini rate-limited this request (HTTP 429). The agent retried with backoff "
            "and still got 429. Wait a minute and try again. "
            "See https://ai.google.dev/gemini-api/docs/rate-limits"
        )

    if code in {401, 403} or "api key not valid" in lower or "permission denied" in lower:
        return LLMError(
            "Gemini rejected the API key. Check GEMINI_API_KEY at "
            "https://aistudio.google.com/apikey"
        )

    if code in {500, 502, 503, 504}:
        return RateLimitError(
            f"Gemini returned a transient server error ({code}). Wait a moment and retry, "
            "or let memento fall back to another free-tier model."
        )

    short = _first_line(message)
    return LLMError(f"Gemini request failed: {short or type(exc).__name__}")


def _grounding_help() -> str:
    return (
        "Gemini Google Search grounding is not available on the free tier for Gemini 3.x "
        "(a 429 mentioning plan/billing is the usual symptom). Keep memento free: use "
        "Tavily search (default) with TAVILY_API_KEY from https://app.tavily.com "
        "(Researcher plan, 1,000 credits/month, no card). Gemini grounding remains an "
        "optional provider if you enable billing: --search-provider gemini"
    )


def _looks_like_grounding_unavailable(lower: str) -> bool:
    if any(marker in lower for marker in _GROUNDING_UNAVAILABLE):
        return True
    return "google_search" in lower and ("not available" in lower or "not supported" in lower)


def _looks_like_model_unavailable(code: object, lower: str) -> bool:
    if code == 404:
        return True
    if "no longer available to new users" in lower:
        return True
    if "this model" in lower and "not available" in lower:
        return True
    return False


def _looks_like_daily_quota(blob: str) -> bool:
    compact = blob.replace(" ", "").replace("-", "").replace("_", "").lower()
    if any(marker.replace(" ", "") in compact for marker in _DAILY_MARKERS):
        return True
    if "limit: 0" in blob or "quota value: 0" in blob:
        return True
    return False


def _error_text(exc: BaseException) -> str:
    """Single copy of the error message — do not concatenate str(exc)+message+details."""
    msg = getattr(exc, "message", None)
    if isinstance(msg, str) and msg.strip():
        return msg.strip()
    return str(exc).strip()


def _first_line(text: str, *, limit: int = 240) -> str:
    line = text.split("\n", 1)[0].strip()
    if len(line) > limit:
        return line[: limit - 1] + "…"
    return line


def _retry_delay_seconds(exc: BaseException, *, fallback: float) -> float:
    blob = _error_text(exc)
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
