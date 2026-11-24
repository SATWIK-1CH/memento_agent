from types import SimpleNamespace

from memento_agent.config import Settings
from memento_agent.llm import GeminiLLM, _strip_fences
from memento_agent.types import ResearchPlan, SourceType


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
            return SimpleNamespace(text=__import__("json").dumps(payload), candidates=[])

    class DummyClient:
        models = DummyModels()

    settings = Settings(api_key="k", db_path=tmp_path / "t.db", search_delay_s=0, fetch_delay_s=0)
    llm = GeminiLLM(settings, client=DummyClient())
    plan = llm.plan("Agonda hotels. Search blogs and Reddit.", [SourceType.BLOG, SourceType.REDDIT])
    assert isinstance(plan, ResearchPlan)
    assert len(plan.sub_queries) == 2
    assert plan.sub_queries[1].source_type is SourceType.REDDIT
