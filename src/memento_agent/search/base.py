"""Search provider interface. Gemini grounding is the default; Tavily/Brave can plug in later."""

from __future__ import annotations

from typing import Protocol

from memento_agent.types import SearchHit, SourceType


class SearchProvider(Protocol):
    """Minimal search backend. Implementations must not write the final answer."""

    name: str

    def search(
        self,
        query: str,
        *,
        source_type: SourceType | None = None,
        max_results: int = 8,
    ) -> list[SearchHit]:
        """Return ranked hits for a single query. May be empty if nothing was found."""
        ...
