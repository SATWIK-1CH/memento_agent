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


def test_youtube_watch_feeds_corroboration(settings, store) -> None:
    from memento_agent.types import VideoWatchNotes
    from memento_agent.youtube import YouTubeAnalyzer

    class YtSearch:
        def search(self, query, *, source_type=None, max_results=8):
            return [
                SearchHit(
                    url="https://www.youtube.com/watch?v=agonda1",
                    title="Agonda beach hotels tour",
                    snippet="Simrose and Palolem day trip",
                    source_type=SourceType.YOUTUBE,
                    query=query,
                )
            ]

    class YtFetcher:
        def fetch(self, hit, *, source_id, iteration):
            return SourceRecord(
                id=source_id,
                url=hit.url,
                title=hit.title,
                source_type=SourceType.YOUTUBE,
                snippet=hit.snippet,
                fetch_status=FetchStatus.YOUTUBE_META,
                fetch_note="title only",
                excerpt=hit.snippet,
                query=hit.query,
                iteration=iteration,
            )

        def close(self) -> None:
            return None

    class WatchLLM(FakeLLM):
        def watch_video(self, url, *, title=""):
            return VideoWatchNotes(
                summary="Tour of Agonda beach huts.",
                stays_shown=["Simrose"],
                prices_stated=["4000 rupees a night"],
                timestamps=["1:20 Simrose"],
            )

        def plan(self, question, source_types):
            from memento_agent.types import PlannedQuery, ResearchPlan

            self.plan_calls += 1
            return ResearchPlan(
                restated_question=question,
                source_types=source_types,
                sub_queries=[
                    PlannedQuery(query="Agonda youtube", source_type=SourceType.YOUTUBE, rationale="vlogs")
                ],
            )

        def judge(self, question, sources, source_types, *, iteration, max_iterations, **kwargs):
            self.judge_calls += 1
            watched = [s for s in sources if s.fetch_status is FetchStatus.YOUTUBE_WATCHED]
            assert watched, "judge must see watched YouTube evidence"
            return Judgment(enough=True, rationale="Watched a relevant vlog.")

    llm = WatchLLM()
    analyzer = YouTubeAnalyzer(llm, max_videos=1, transcript_fetcher=lambda _vid: "")
    agent = ResearchAgent(
        settings,
        llm=llm,
        search=YtSearch(),
        fetcher=YtFetcher(),
        store=store,
        youtube=analyzer,
        sleeper=lambda _d: None,
    )
    result = agent.run("Agonda hotels. Search YouTube.", source_types=[SourceType.YOUTUBE])
    assert any(s.fetch_status is FetchStatus.YOUTUBE_WATCHED for s in result.sources)
    assert "watched" in result.markdown.lower()


def test_reddit_extract_is_logged(settings, store) -> None:
    class ExtractSearch(FakeSearch):
        def extract(self, urls, query=""):
            return {urls[0]: "Longer extracted Reddit thread about Simrose on the sand. " * 8}

    events: list[str] = []
    agent = ResearchAgent(
        settings,
        llm=FakeLLM(),
        search=ExtractSearch(),
        fetcher=FakeFetcher(),
        store=store,
        on_event=lambda e: events.append(e.message),
        sleeper=lambda _d: None,
    )
    agent.run(
        "Best hotels in Agonda. Search blogs and Reddit.",
        source_types=[SourceType.BLOG, SourceType.REDDIT],
    )
    joined = " ".join(events).lower()
    assert "tavily extract" in joined


def test_empty_question_rejected(settings, store) -> None:
    agent = ResearchAgent(settings, llm=FakeLLM(), search=FakeSearch(), fetcher=FakeFetcher(), store=store)
    try:
        agent.run("  ")
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_judge_503_uses_heuristic_instead_of_failing(settings, store) -> None:
    from memento_agent.errors import RateLimitError

    class FlakyJudge(FakeLLM):
        def judge(self, *args, **kwargs):
            self.judge_calls += 1
            raise RateLimitError("Gemini returned a transient server error (503).")

    events: list[str] = []
    agent = ResearchAgent(
        settings,
        llm=FlakyJudge(),
        search=FakeSearch(),
        fetcher=FakeFetcher(),
        store=store,
        on_event=lambda e: events.append(e.kind),
        sleeper=lambda _d: None,
    )
    result = agent.run(
        "Best hotels in Agonda. Search blogs and Reddit.",
        source_types=[SourceType.BLOG, SourceType.REDDIT],
    )
    assert result.markdown
    assert "fallback" in events
    assert store.get_run(result.run_id)["status"] == "completed"
