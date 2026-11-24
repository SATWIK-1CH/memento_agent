"""Regression: live Agonda run (24 Nov 2026) dropped named stays and mis-cited Cuba."""

from __future__ import annotations

from tests.fakes import FakeLLM

from memento_agent.agent import ResearchAgent
from memento_agent.types import (
    AnswerSection,
    FetchStatus,
    Judgment,
    PlannedQuery,
    RankedItem,
    ResearchAnswer,
    ResearchPlan,
    SearchHit,
    SourceRecord,
    SourceType,
)


HOTELS = [
    "Agonda Villas",
    "The Bay Agonda",
    "Simrose",
    "The Rose",
    "Jardim A Mar",
    "O2H",
    "Om Sai",
    "Oasis Beach Resort",
]


class FixtureSearch:
    def search(self, query, *, source_type=None, max_results=8):
        if source_type is SourceType.YOUTUBE:
            return [
                SearchHit(
                    url="https://www.youtube.com/watch?v=oasis",
                    title="Oasis Beach Resort Agonda tour",
                    snippet="Oasis Beach Resort walkthrough",
                    source_type=SourceType.YOUTUBE,
                    query=query,
                ),
                SearchHit(
                    url="https://www.youtube.com/watch?v=hindi",
                    title="Agonda beach vlog",
                    snippet="Generic Hindi walk along Agonda beach. No hotel names.",
                    source_type=SourceType.YOUTUBE,
                    query=query,
                ),
            ]
        if source_type is SourceType.REDDIT:
            return [
                SearchHit(
                    url="https://www.reddit.com/r/goatravel/comments/abc/agonda/",
                    title="Agonda in December?",
                    snippet="We stayed at Simrose — mid-range, on the sand.",
                    source_type=SourceType.REDDIT,
                    query=query,
                )
            ]
        if source_type is SourceType.BLOG:
            return [
                SearchHit(
                    url="https://blog.example/first-timers-agonda",
                    title="Agonda first-timers guide",
                    snippet="Jardim A Mar and O2H. Om Sai. Cuba Agonda.",
                    source_type=SourceType.BLOG,
                    query=query,
                ),
                SearchHit(
                    url="https://www.tripadvisor.com/agonda",
                    title="Om Sai Agonda",
                    snippet="Om Sai is a simple stay. Rooms from INR 2500.",
                    source_type=SourceType.BLOG,
                    query=query,
                ),
            ]
        return [
            SearchHit(
                url="https://www.booking.com/agonda",
                title="Agonda hotels",
                snippet="list",
                source_type=SourceType.WEB,
                query=query,
            )
        ]


class FixtureFetcher:
    def fetch(self, hit, *, source_id, iteration):
        pages = {
            "booking.com": SourceRecord(
                id=source_id,
                url=hit.url,
                title="Agonda hotels — Booking.com",
                source_type=SourceType.WEB,
                fetch_status=FetchStatus.OK,
                excerpt=(
                    "Places to stay: Agonda Villas, The Bay Agonda, Simrose, The Rose. "
                    "Simrose from ₹4,500 /night."
                ),
                query=hit.query,
                iteration=iteration,
            ),
            "reddit.com": SourceRecord(
                id=source_id,
                url=hit.url,
                title=hit.title,
                source_type=SourceType.REDDIT,
                snippet=hit.snippet,
                fetch_status=FetchStatus.BLOCKED,
                fetch_note="HTTP 403 from https://old.reddit.com/r/goatravel/comments/abc/agonda",
                excerpt=hit.snippet,
                query=hit.query,
                iteration=iteration,
            ),
            "oasis": SourceRecord(
                id=source_id,
                url=hit.url,
                title="Oasis Beach Resort Agonda tour",
                source_type=SourceType.YOUTUBE,
                fetch_status=FetchStatus.YOUTUBE_META,
                excerpt="Channel: TravelChan",
                snippet=hit.snippet,
                query=hit.query,
                iteration=iteration,
            ),
            "hindi": SourceRecord(
                id=source_id,
                url=hit.url,
                title="Agonda beach vlog",
                source_type=SourceType.YOUTUBE,
                fetch_status=FetchStatus.YOUTUBE_META,
                excerpt="Generic Hindi walk along Agonda beach. No hotel names.",
                snippet=hit.snippet,
                query=hit.query,
                iteration=iteration,
            ),
            "tripadvisor.com": SourceRecord(
                id=source_id,
                url=hit.url,
                title="Om Sai Agonda",
                source_type=SourceType.BLOG,
                fetch_status=FetchStatus.OK,
                excerpt="Om Sai is a simple stay. 4.2/5. Rooms from INR 2500.",
                query=hit.query,
                iteration=iteration,
            ),
        }
        for key, rec in pages.items():
            if key in hit.url:
                return rec
        return SourceRecord(
            id=source_id,
            url=hit.url,
            title="Agonda first-timers guide",
            source_type=SourceType.BLOG,
            fetch_status=FetchStatus.OK,
            excerpt=(
                "We liked Jardim A Mar and O2H. Palolem and Patnem are easy day trips. "
                "Cola is quieter. Om Sai is a simple stay. Agonda Cottages are wood and "
                "coconut palm cottages on the sand. Cuba Agonda has a pool."
            ),
            query=hit.query,
            iteration=iteration,
        )

    def close(self) -> None:
        return None


class SloppyWriter(FakeLLM):
    def plan(self, question, source_types):
        self.plan_calls += 1
        return ResearchPlan(
            restated_question=question,
            source_types=source_types,
            sub_queries=[
                PlannedQuery(query="agonda hotels booking", source_type=SourceType.WEB),
                PlannedQuery(query="agonda first timers blog", source_type=SourceType.BLOG),
                PlannedQuery(query="agonda youtube", source_type=SourceType.YOUTUBE),
                PlannedQuery(query="agonda reddit", source_type=SourceType.REDDIT),
            ],
        )

    def judge(self, question, sources, source_types, *, iteration, max_iterations, **kwargs):
        self.judge_calls += 1
        return Judgment(
            enough=True,
            rationale="Looks fine after one pass.",
            gaps=[],
            missing_source_types=[SourceType.REDDIT],
            entity_checks=[],
        )

    def write(self, question, sources, source_types, judgment, **kwargs):
        self.write_calls += 1
        return ResearchAnswer(
            summary="A few stays in Agonda.",
            sections=[
                AnswerSection(
                    title="Hotels in Agonda",
                    items=[
                        RankedItem(
                            name="Cuba Agonda",
                            why="wood and coconut palm cottages",
                            source_ids=["S2", "S4"],
                        )
                    ],
                )
            ],
            limitations=["Needs more research"],
            open_questions=["Needs more research", "What is Simrose charging this season?"],
        )


def test_answer_keeps_named_hotels_and_does_not_misattribute(settings, store) -> None:
    settings = settings.__class__(
        **{**settings.__dict__, "max_iterations": 2, "max_searches": 8, "max_pages_per_search": 4}
    )
    llm = SloppyWriter()
    agent = ResearchAgent(
        settings,
        llm=llm,
        search=FixtureSearch(),
        fetcher=FixtureFetcher(),
        store=store,
        sleeper=lambda _d: None,
    )
    result = agent.run(
        "Best hotels in Agonda, Goa. Search YouTube, travel blogs and Reddit.",
        source_types=[SourceType.YOUTUBE, SourceType.REDDIT, SourceType.BLOG, SourceType.WEB],
    )
    lower = result.markdown.lower()
    for name in HOTELS:
        assert name.lower() in lower, name
    assert "patnem" in lower
    assert "cola" in lower
    assert "4,500" in result.markdown or "4500" in lower or "₹" in result.markdown
    cuba = None
    for section in result.answer.sections:
        for item in section.items:
            if "cuba" in item.name.lower():
                cuba = item
    assert cuba is not None
    # S2 in the sloppy writer was the youtube/hindi or cottages page depending on fetch order;
    # after scrub, Cuba may only keep sources that actually mention Cuba.
    for sid in cuba.source_ids:
        src = next(s for s in result.sources if s.id == sid)
        blob = f"{src.title} {src.excerpt} {src.snippet}".lower()
        assert "cuba" in blob
    assert "8 reddit" not in lower
    assert "needs more research" not in lower
    assert llm.judge_calls >= 1
    # One-pass enough=true must not stop while stays are single-type (max_iterations=2).
    assert result.iterations >= 2
