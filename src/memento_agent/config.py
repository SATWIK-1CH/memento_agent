"""Runtime settings from env, .env, and CLI flags."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from memento_agent.errors import ConfigError

# New free-tier Gemini keys cannot call gemini-2.5-* (404 "no longer available to
# new users") even though models.list() still lists them. gemini-3.5-flash-lite
# generate works; gemini-3.8-flash is a fallback on 503/404. Google Search
# grounding is not available on free 3.x — search defaults to Tavily.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_FALLBACK_MODELS = ("gemini-3.8-flash",)
DEFAULT_SEARCH_PROVIDER = "tavily"
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_MAX_SEARCHES = 8
DEFAULT_MAX_PAGES_PER_SEARCH = 4
DEFAULT_MAX_VIDEOS = 2
DEFAULT_EXCERPT_CHARS = 8_000


def default_db_path() -> Path:
    override = os.environ.get("MEMENTO_DB_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".memento" / "memento.db"


def load_env_files() -> None:
    """Load a local .env if present, without overriding real environment vars."""
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str = DEFAULT_MODEL
    fallback_models: tuple[str, ...] = DEFAULT_FALLBACK_MODELS
    search_provider: str = DEFAULT_SEARCH_PROVIDER
    tavily_api_key: str = ""
    search_model: str | None = None
    db_path: Path = Path.home() / ".memento" / "memento.db"
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_searches: int = DEFAULT_MAX_SEARCHES
    max_pages_per_search: int = DEFAULT_MAX_PAGES_PER_SEARCH
    max_videos: int = DEFAULT_MAX_VIDEOS
    excerpt_chars: int = DEFAULT_EXCERPT_CHARS
    request_timeout_s: float = 20.0
    search_delay_s: float = 0.4
    fetch_delay_s: float = 0.2
    skip_preflight_probe: bool = False

    @property
    def effective_search_model(self) -> str:
        return self.search_model or self.model

    @property
    def search_label(self) -> str:
        if self.search_provider == "gemini":
            return f"gemini-google-search:{self.effective_search_model}"
        return "tavily"

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str | None = None,
        model: str | None = None,
        search_model: str | None = None,
        search_provider: str | None = None,
        tavily_api_key: str | None = None,
        db_path: Path | str | None = None,
        max_iterations: int | None = None,
        max_searches: int | None = None,
        max_pages_per_search: int | None = None,
        max_videos: int | None = None,
        load_dotenv_file: bool = True,
        require_key: bool = True,
    ) -> Settings:
        if load_dotenv_file:
            load_env_files()
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key and require_key:
            raise ConfigError(
                "No Gemini API key found. Set GEMINI_API_KEY (or put it in a local .env). "
                "Create a free key at https://aistudio.google.com/apikey"
            )
        path = Path(db_path) if db_path else default_db_path()
        provider = (
            search_provider
            or os.environ.get("MEMENTO_SEARCH_PROVIDER")
            or DEFAULT_SEARCH_PROVIDER
        ).strip().lower()
        if provider in {"google", "gemini-google-search", "grounding"}:
            provider = "gemini"
        if provider not in {"tavily", "gemini"}:
            raise ConfigError(
                f"Unknown search provider {provider!r}. Use 'tavily' (default, free) or 'gemini' "
                "(Google Search grounding; requires Gemini billing)."
            )
        fallback_raw = os.environ.get("MEMENTO_FALLBACK_MODELS")
        fallbacks = (
            tuple(m.strip() for m in fallback_raw.split(",") if m.strip())
            if fallback_raw
            else DEFAULT_FALLBACK_MODELS
        )
        skip_probe = os.environ.get("MEMENTO_SKIP_PREFLIGHT", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        return cls(
            api_key=key or "",
            model=model or os.environ.get("MEMENTO_MODEL") or DEFAULT_MODEL,
            fallback_models=fallbacks,
            search_provider=provider,
            tavily_api_key=tavily_api_key
            if tavily_api_key is not None
            else (os.environ.get("TAVILY_API_KEY") or ""),
            search_model=search_model or os.environ.get("MEMENTO_SEARCH_MODEL") or None,
            db_path=path,
            max_iterations=_int_env("MEMENTO_MAX_ITERATIONS", max_iterations, DEFAULT_MAX_ITERATIONS),
            max_searches=_int_env("MEMENTO_MAX_SEARCHES", max_searches, DEFAULT_MAX_SEARCHES),
            max_pages_per_search=_int_env(
                "MEMENTO_MAX_PAGES", max_pages_per_search, DEFAULT_MAX_PAGES_PER_SEARCH
            ),
            max_videos=_int_env("MEMENTO_MAX_VIDEOS", max_videos, DEFAULT_MAX_VIDEOS),
            skip_preflight_probe=skip_probe,
        )


def _int_env(name: str, explicit: int | None, default: int) -> int:
    if explicit is not None:
        return explicit
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
