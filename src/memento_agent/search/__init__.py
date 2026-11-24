from memento_agent.config import Settings
from memento_agent.errors import ConfigError
from memento_agent.search.base import SearchProvider
from memento_agent.search.gemini import GeminiSearchProvider
from memento_agent.search.tavily import TavilySearchProvider

__all__ = [
    "SearchProvider",
    "GeminiSearchProvider",
    "TavilySearchProvider",
    "build_search_provider",
]


def build_search_provider(settings: Settings, llm: object) -> SearchProvider:
    if settings.search_provider == "gemini":
        return GeminiSearchProvider(llm)
    if settings.search_provider == "tavily":
        return TavilySearchProvider(api_key=settings.tavily_api_key)
    raise ConfigError(
        f"Unknown search provider {settings.search_provider!r}. Use tavily or gemini."
    )
