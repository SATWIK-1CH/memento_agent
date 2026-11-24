import json
from types import SimpleNamespace

from memento_agent.config import Settings
from memento_agent.errors import ModelUnavailableError
from memento_agent.llm import GeminiLLM, _strip_fences
from memento_agent.types import ResearchPlan, SourceType, VideoWatchNotes


def test_strip_fences() -> None:
    raw = '```json\n{"restated_question": "q", "source_types": ["web"], "sub_queries": []}\n```'
    assert _strip_fences(raw).startswith("{")


def test_plan_parses_structured_response(tmp_path) -> None:
    payload = {
        "restated_question": "hotels in Agonda",
        "source_types": ["blog", "reddit"],
        "sub_queries": [
            {"query": "Agonda hotels", "source_type": "blog", "rationale": "blogs"},
            {"query": "Agonda site:reddit.com/r/goatravel", "source_type": "reddit", "rationale": "reddit"},
        ],
    }

    class DummyModels:
        def generate_content(self, **kwargs):
            assert kwargs["config"].response_mime_type == "application/json"
            return SimpleNamespace(text=json.dumps(payload), candidates=[])

    class DummyClient:
        models = DummyModels()

    settings = Settings(api_key="k", db_path=tmp_path / "t.db", search_delay_s=0, fetch_delay_s=0)
    llm = GeminiLLM(settings, client=DummyClient())
    plan = llm.plan("Agonda hotels. Search blogs and Reddit.", [SourceType.BLOG, SourceType.REDDIT])
    assert isinstance(plan, ResearchPlan)
    assert len(plan.sub_queries) == 2
    assert plan.sub_queries[1].source_type is SourceType.REDDIT


def test_falls_back_on_404(tmp_path) -> None:
    calls: list[str] = []

    class DummyModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "gemini-3.5-flash-lite":
                err = RuntimeError("gone")
                err.code = 404  # type: ignore[attr-defined]
                err.message = "This model models/gemini-3.5-flash-lite is no longer available to new users."  # type: ignore[attr-defined]
                raise err
            payload = {
                "restated_question": "q",
                "source_types": ["web"],
                "sub_queries": [{"query": "Agonda", "source_type": "web", "rationale": "x"}],
            }
            return SimpleNamespace(text=json.dumps(payload), candidates=[])

    class DummyClient:
        models = DummyModels()

    settings = Settings(
        api_key="k",
        model="gemini-3.5-flash-lite",
        fallback_models=("gemini-3.8-flash",),
        db_path=tmp_path / "t.db",
    )
    notes: list[str] = []
    llm = GeminiLLM(settings, client=DummyClient(), on_fallback=notes.append)
    plan = llm.plan("Agonda", [SourceType.WEB])
    assert plan.sub_queries
    assert calls == ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    assert llm.model == "gemini-3.8-flash"
    assert notes and "gemini-3.8-flash" in notes[0]


def test_watch_video_parses_notes(tmp_path) -> None:
    notes = {
        "summary": "Beach walk",
        "stays_shown": ["Simrose"],
        "prices_stated": [],
        "pros": [],
        "cons": [],
        "timestamps": ["1:00 beach"],
        "caveats": [],
    }

    class DummyModels:
        def generate_content(self, **kwargs):
            contents = kwargs["contents"]
            assert contents
            return SimpleNamespace(text=json.dumps(notes), candidates=[])

    class DummyClient:
        models = DummyModels()

    settings = Settings(api_key="k", db_path=tmp_path / "t.db")
    llm = GeminiLLM(settings, client=DummyClient())
    result = llm.watch_video("https://www.youtube.com/watch?v=abc", title="Agonda")
    assert isinstance(result, VideoWatchNotes)
    assert result.stays_shown == ["Simrose"]


def test_search_does_not_replace_generation_model(tmp_path) -> None:
    class DummyModels:
        def generate_content(self, **kwargs):
            assert kwargs["model"] == "search-model"
            return SimpleNamespace(text="ok", candidates=[])

    class DummyClient:
        models = DummyModels()

    settings = Settings(
        api_key="k",
        model="gen-model",
        search_model="search-model",
        db_path=tmp_path / "t.db",
    )
    llm = GeminiLLM(settings, client=DummyClient())
    llm.generate_search("Agonda hotels")
    assert llm.model == "gen-model"


def test_falls_back_on_503_and_logs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("memento_agent.retry.time.sleep", lambda _d: None)
    calls: list[str] = []

    class DummyModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "gemini-3.5-flash-lite":
                err = RuntimeError("The model is overloaded")
                err.code = 503  # type: ignore[attr-defined]
                err.status = "UNAVAILABLE"  # type: ignore[attr-defined]
                raise err
            payload = {
                "restated_question": "q",
                "source_types": ["web"],
                "sub_queries": [{"query": "Agonda", "source_type": "web", "rationale": "x"}],
            }
            return SimpleNamespace(text=json.dumps(payload), candidates=[])

    class DummyClient:
        models = DummyModels()

    settings = Settings(
        api_key="k",
        model="gemini-3.5-flash-lite",
        fallback_models=("gemini-3.8-flash",),
        db_path=tmp_path / "t.db",
    )
    notes: list[str] = []
    llm = GeminiLLM(settings, client=DummyClient(), on_fallback=notes.append)
    plan = llm.plan("Agonda", [SourceType.WEB])
    assert plan.sub_queries
    assert calls[0] == "gemini-3.5-flash-lite"
    assert "gemini-3.8-flash" in calls
    assert notes
    assert "gemini-3.8-flash" in notes[0]


def test_video_part_uses_static_processing() -> None:
    from google.genai import types

    from memento_agent.llm import _video_part

    part = _video_part(types, "https://www.youtube.com/watch?v=abc")
    processing = getattr(part, "media_processing", None)
    assert processing in {types.MediaProcessing.STATIC, "STATIC"}


def test_all_models_unavailable(tmp_path) -> None:
    class DummyModels:
        def generate_content(self, **kwargs):
            err = RuntimeError("gone")
            err.code = 404  # type: ignore[attr-defined]
            err.message = "This model is no longer available to new users."  # type: ignore[attr-defined]
            raise err

    class DummyClient:
        models = DummyModels()

    settings = Settings(
        api_key="k",
        model="gemini-3.5-flash-lite",
        fallback_models=("gemini-3.8-flash",),
        db_path=tmp_path / "t.db",
    )
    llm = GeminiLLM(settings, client=DummyClient())
    try:
        llm.plan("Agonda", [SourceType.WEB])
        raise AssertionError("should have raised")
    except ModelUnavailableError:
        pass
