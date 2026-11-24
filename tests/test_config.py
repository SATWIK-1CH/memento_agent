from memento_agent.config import DEFAULT_MODEL, DEFAULT_SEARCH_PROVIDER, Settings


def test_defaults_are_free_tier() -> None:
    assert DEFAULT_MODEL == "gemini-3.5-flash-lite"
    assert DEFAULT_SEARCH_PROVIDER == "tavily"


def test_from_env_tavily(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    monkeypatch.delenv("MEMENTO_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("MEMENTO_MODEL", raising=False)
    settings = Settings.from_env(db_path=tmp_path / "x.db", load_dotenv_file=False)
    assert settings.model == "gemini-3.5-flash-lite"
    assert settings.search_provider == "tavily"
    assert settings.tavily_api_key == "t"
    assert settings.search_label == "tavily"
    assert settings.max_videos == 2
