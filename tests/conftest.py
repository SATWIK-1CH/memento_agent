from __future__ import annotations

from pathlib import Path

import pytest

from memento_agent.config import Settings
from memento_agent.store import Store


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test-key",
        model="gemini-2.5-flash",
        db_path=tmp_path / "memento.db",
        max_iterations=3,
        max_searches=8,
        max_pages_per_search=3,
        search_delay_s=0.0,
        fetch_delay_s=0.0,
    )


@pytest.fixture
def store(settings: Settings) -> Store:
    return Store(settings.db_path)
