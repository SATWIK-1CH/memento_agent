from memento_agent.config import Settings
from memento_agent.errors import ConfigError, ModelUnavailableError
from memento_agent.preflight import preflight


def _settings(**kwargs) -> Settings:
    values = dict(
        api_key="g",
        tavily_api_key="t",
        search_provider="tavily",
        skip_preflight_probe=True,
    )
    values.update(kwargs)
    return Settings(**values)


def test_missing_gemini_key() -> None:
    try:
        preflight(_settings(api_key=""))
        raise AssertionError("should have raised")
    except ConfigError as exc:
        assert "GEMINI_API_KEY" in str(exc)


def test_missing_tavily_key_explains_grounding() -> None:
    try:
        preflight(_settings(tavily_api_key=""))
        raise AssertionError("should have raised")
    except ConfigError as exc:
        text = str(exc)
        assert "TAVILY_API_KEY" in text
        assert "billing" in text.lower() or "grounding" in text.lower()


def test_gemini_provider_does_not_require_tavily() -> None:
    preflight(_settings(search_provider="gemini", tavily_api_key=""))


def test_model_probe_failure_becomes_config_error() -> None:
    class Boom:
        def probe(self) -> str:
            raise ModelUnavailableError("gemini-2.5-flash is gone")

    try:
        preflight(_settings(skip_preflight_probe=False), llm=Boom())
        raise AssertionError("should have raised")
    except ConfigError as exc:
        assert "gone" in str(exc)
