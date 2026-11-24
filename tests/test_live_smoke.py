"""Optional live smoke against Gemini. Skipped when GEMINI_API_KEY is unset."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

EXAMPLE = (
    "Best hotels in Agonda, Goa, and which other South Goa areas are best to cover "
    "on a trip. Search YouTube, travel blogs and Reddit."
)


@pytest.mark.live
def test_live_agonda_smoke(tmp_path: Path) -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        pytest.skip("GEMINI_API_KEY not set; live smoke was not run")

    from memento_agent.agent import research
    from memento_agent.config import Settings

    settings = Settings.from_env(
        db_path=tmp_path / "live.db",
        max_iterations=2,
        max_searches=4,
        max_pages_per_search=2,
        load_dotenv_file=True,
    )
    settings = Settings(
        api_key=settings.api_key,
        model=settings.model,
        search_model=settings.search_model,
        db_path=settings.db_path,
        max_iterations=2,
        max_searches=4,
        max_pages_per_search=2,
        search_delay_s=0.5,
        fetch_delay_s=0.2,
    )
    result = research(EXAMPLE, settings=settings)
    assert result.markdown
    assert result.sources
    assert result.searches_used >= 1
    lower = result.markdown.lower()
    assert "source" in lower
    # A useful answer mentions hotels/areas and is not just an error dump.
    assert any(word in lower for word in ("hotel", "agonda", "palolem", "patnem", "goa"))
