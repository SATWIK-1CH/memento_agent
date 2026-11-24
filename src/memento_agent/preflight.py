"""Startup checks so the user sees a clear gap *before* the plan step."""

from __future__ import annotations

from typing import Any

from memento_agent.config import Settings
from memento_agent.errors import ConfigError, GroundingUnavailableError, ModelUnavailableError
from memento_agent.events import EventBus


def preflight(settings: Settings, llm: Any | None = None, *, events: EventBus | None = None) -> None:
    """Raise ConfigError with every missing piece; warn on optional Gemini grounding."""
    problems: list[str] = []

    if not settings.api_key:
        problems.append(
            "Missing GEMINI_API_KEY. Get a free key at https://aistudio.google.com/apikey"
        )

    if settings.search_provider == "tavily" and not settings.tavily_api_key:
        problems.append(
            "Missing TAVILY_API_KEY. Web search no longer uses Gemini grounding on the free "
            "tier (Gemini 3.x returns 429 plan/billing). Tavily's Researcher plan is free: "
            "1,000 credits/month, basic search = 1 credit, 100 RPM, no credit card. "
            "Create a key at https://app.tavily.com — see https://docs.tavily.com/documentation/api-credits"
        )

    if settings.search_provider == "gemini":
        warning = (
            "Search provider is Gemini Google Search grounding. That tool is not available "
            "on the Gemini API free tier for 3.x models (429 RESOURCE_EXHAUSTED mentioning "
            "plan and billing). Enable billing, or switch to Tavily: "
            "MEMENTO_SEARCH_PROVIDER=tavily and TAVILY_API_KEY."
        )
        if events:
            events.emit("preflight", warning)
        else:
            # Opt-in provider: do not hard-fail; the first grounded call will.
            pass

    if problems:
        raise ConfigError("Cannot start research:\n- " + "\n- ".join(problems))

    if llm is not None and not settings.skip_preflight_probe:
        _probe_model(llm, events)


def _probe_model(llm: Any, events: EventBus | None) -> None:
    if not hasattr(llm, "probe"):
        return
    try:
        model = llm.probe()
        if events and model:
            events.emit("preflight", f"Gemini model ready: {model}")
    except ModelUnavailableError as exc:
        raise ConfigError(str(exc)) from exc
    except GroundingUnavailableError as exc:
        raise ConfigError(str(exc)) from exc
