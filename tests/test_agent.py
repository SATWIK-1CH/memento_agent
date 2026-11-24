from __future__ import annotations

from tests.fakes import FakeFetcher, FakeLLM, FakeSearch

from memento_agent.agent import ResearchAgent, research
from memento_agent.errors import QuotaExhaustedError
from memento_agent.types import (
    FetchStatus,
    Judgment,
    SearchHit,
    SourceRecord,
    SourceType,
)


def test_loop_searches_again_when_judge_finds_gaps(settings, store) -> None:
    llm = FakeLLM()
    search = FakeSearch()
    fetcher = FakeFetcher()
    agent = ResearchAgent(settings, llm=llm, search=search, fetcher=fetcher, store=store, sleeper=lambda _d: None)
    result = agent.run(
        "Best hotels in Agonda, Goa. Search travel blogs and Reddit.",
        source_types=[SourceType.BLOG, SourceType.REDDIT],
    )
    assert llm.plan_calls == 1
    assert llm.judge_calls == 2
    assert llm.write_calls == 1
    assert result.iterations == 2
    assert result.searches_used >= 2
    assert any(s.source_type is SourceType.REDDIT for s in result.sources)
    assert any(s.fetch_status is FetchStatus.DEAD for s in result.sources)
    assert "parked" in result.markdown.lower() or "dead" in result.markdown.lower()
    row = store.get_run(result.run_id)
    assert row is not None
    assert row["status"] == "completed"
    assert store.get_sub_queries(result.run_id)


def test_research_helper_wires_fakes(settings, store) -> None:
    result = research(
        "Agonda hotels. Search blogs and Reddit.",
        settings=settings,
        store=store,
        llm=FakeLLM(),
        search=FakeSearch(),
        fetcher=FakeFetcher(),
    )
    assert result.markdown
    assert result.answer.sections


def test_failed_run_is_persisted(settings, store) -> None:
    class BoomSearch:
        def search(self, *args, **kwargs):
            raise QuotaExhaustedError("Gemini free-tier quota is exhausted for today.")

    agent = ResearchAgent(
        settings,
        llm=FakeLLM(),
        search=BoomSearch(),
        fetcher=FakeFetcher(),
        store=store,
        sleeper=lambda _d: None,
    )
    try:
        agent.run("Agonda?", source_types=[SourceType.WEB])
        raise AssertionError("should have raised")
    except QuotaExhaustedError:
        pass
    rows = store.list_runs()
    assert rows[0]["status"] == "failed"


def test_failed_empty_hits_skip_the_writer(settings, store) -> None:
    class EmptySearch:
        def search(self, *args, **kwargs):
            return [
                SearchHit(
                    url="https://example.com/blocked",
                    title="",
                    snippet="",
                    source_type=SourceType.WEB,
                )
            ]

    class EmptyFetcher:
        def fetch(self, hit, *, source_id, iteration):
            return SourceRecord(
                id=source_id,
                url=hit.url,
                source_type=SourceType.WEB,
                fetch_status=FetchStatus.FAILED,
                fetch_note="timeout",
                excerpt="",
                snippet="",
            )

        def close(self) -> None:
            return None

    class StopLLM(FakeLLM):
        def judge(self, *args, **kwargs):
            self.judge_calls += 1
            return Judgment(enough=True, rationale="nothing usable")

    llm = StopLLM()
    agent = ResearchAgent(
        settings,
        llm=llm,
        search=EmptySearch(),
        fetcher=EmptyFetcher(),
        store=store,
        sleeper=lambda _d: None,
    )
    result = agent.run("Agonda hotels?", source_types=[SourceType.WEB])
    assert llm.write_calls == 0
    assert "No citable sources" in result.markdown


def test_empty_question_rejected(settings, store) -> None:
    agent = ResearchAgent(settings, llm=FakeLLM(), search=FakeSearch(), fetcher=FakeFetcher(), store=store)
    try:
        agent.run("  ")
        raise AssertionError("should have raised")
    except ValueError:
        pass
