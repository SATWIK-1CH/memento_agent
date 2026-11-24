from __future__ import annotations

import memento_agent._warnings  # noqa: F401

from pathlib import Path

import pytest

from memento_agent.config import Settings
from memento_agent.store import Store


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        fallback_models=("gemini-3.8-flash",),
        search_provider="tavily",
        tavily_api_key="tvly-test",
        db_path=tmp_path / "memento.db",
        max_iterations=3,
        max_searches=8,
        max_pages_per_search=3,
        max_videos=2,
        search_delay_s=0.0,
        fetch_delay_s=0.0,
        skip_preflight_probe=True,
    )


@pytest.fixture
def store(settings: Settings) -> Store:
    return Store(settings.db_path)
