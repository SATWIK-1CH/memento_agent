from __future__ import annotations

from memento_agent.types import (
    AnswerSection,
    EntityCheck,
    FetchStatus,
    FollowupQuery,
    Judgment,
    PlannedQuery,
    RankedItem,
    ResearchAnswer,
    ResearchPlan,
    SearchHit,
    SourceRecord,
    SourceType,
)


class FakeLLM:
    def __init__(self) -> None:
        self.judge_calls = 0
        self.plan_calls = 0
        self.write_calls = 0

    def plan(self, question: str, source_types: list[SourceType]) -> ResearchPlan:
        self.plan_calls += 1
        return ResearchPlan(
            restated_question=question,
            source_types=source_types,
            sub_queries=[
                PlannedQuery(
                    query="best hotels Agonda Goa",
                    source_type=SourceType.BLOG,
                    rationale="listicles and travel blogs",
                )
            ],
        )

    def judge(self, question, sources, source_types, *, iteration, max_iterations) -> Judgment:
        self.judge_calls += 1
        if self.judge_calls == 1:
            return Judgment(
                enough=False,
                rationale="Need Reddit corroboration of hotel names; listicle-only so far.",
                gaps=["reddit mentions of Simrose"],
                missing_source_types=[SourceType.REDDIT],
                entity_checks=[
                    EntityCheck(
                        name="Simrose",
                        kind="hotel",
                        corroborated=False,
                        source_ids=[sources[0].id] if sources else [],
                        source_types=[SourceType.BLOG] if sources else [],
                        notes="Only one travel listicle so far.",
                    )
                ],
                followup_queries=[
                    FollowupQuery(
                        query="Agonda hotels site:reddit.com/r/goatravel",
                        source_type=SourceType.REDDIT,
                        gap="reddit corroboration",
                    )
                ],
            )
        types = list({s.source_type for s in sources})
        return Judgment(
            enough=True,
            rationale="Hotel names now appear in a blog and a Reddit snippet.",
            corroborated_points=["Simrose is a mid-range beach hotel in Agonda"],
            entity_checks=[
                EntityCheck(
                    name="Simrose",
                    kind="hotel",
                    corroborated=len(types) >= 2,
                    source_ids=[s.id for s in sources],
                    source_types=types,
                )
            ],
        )

    def write(self, question, sources, source_types, judgment) -> ResearchAnswer:
        self.write_calls += 1
        ids = [s.id for s in sources if s.citable]
        return ResearchAnswer(
            summary="Agonda has a short beach strip of mid-range huts; Palolem and Patnem are the usual add-ons.",
            sections=[
                AnswerSection(
                    title="Hotels in Agonda",
                    narrative="Simrose is repeatedly mentioned as a mid-range beach hotel [S1].",
                    items=[
                        RankedItem(
                            name="Simrose",
                            tier="mid-range",
                            why="Beach-facing huts; named in travel blogs and traveller comments.",
                            source_ids=ids[:2] or ids,
                        )
                    ],
                )
            ],
            limitations=[],
            open_questions=[],
        )


class FakeSearch:
    def __init__(self) -> None:
        self.queries: list[tuple[str, SourceType | None]] = []

    def search(self, query: str, *, source_type: SourceType | None = None, max_results: int = 8):
        self.queries.append((query, source_type))
        if source_type is SourceType.REDDIT or "reddit" in query.lower():
            return [
                SearchHit(
                    url="https://www.reddit.com/r/goatravel/comments/abc/agonda_hotels/",
                    title="Agonda in December?",
                    snippet="We stayed at Simrose — mid-range, on the sand. Palolem for nightlife.",
                    source_type=SourceType.REDDIT,
                    query=query,
                )
            ]
        return [
            SearchHit(
                url="https://travelblog.example/agonda-hotels",
                title="Best hotels in Agonda",
                snippet="Simrose is a popular mid-range pick on Agonda beach.",
                source_type=SourceType.BLOG,
                query=query,
            ),
            SearchHit(
                url="https://parked-hotel.example/",
                title="Official Simrose site",
                snippet="Book direct",
                source_type=SourceType.WEB,
                query=query,
            ),
        ]


class FakeFetcher:
    def __init__(self) -> None:
        self.fetched: list[str] = []

    def fetch(self, hit: SearchHit, *, source_id: str, iteration: int) -> SourceRecord:
        self.fetched.append(hit.url)
        if "parked" in hit.url:
            return SourceRecord(
                id=source_id,
                url=hit.url,
                title="This domain is for sale",
                source_type=SourceType.WEB,
                snippet=hit.snippet,
                fetch_status=FetchStatus.DEAD,
                fetch_note="URL looks parked or dead",
                excerpt="",
                query=hit.query,
                iteration=iteration,
            )
        if hit.source_type is SourceType.REDDIT:
            return SourceRecord(
                id=source_id,
                url=hit.url,
                title=hit.title,
                source_type=SourceType.REDDIT,
                snippet=hit.snippet,
                fetch_status=FetchStatus.SNIPPET_ONLY,
                fetch_note="Could not open the Reddit thread; used the search snippet only.",
                excerpt=hit.snippet,
                query=hit.query,
                iteration=iteration,
            )
        return SourceRecord(
            id=source_id,
            url=hit.url,
            title=hit.title,
            source_type=hit.source_type,
            snippet=hit.snippet,
            fetch_status=FetchStatus.OK,
            excerpt=hit.snippet + " Palolem and Patnem are the usual day-trip pair.",
            query=hit.query,
            iteration=iteration,
        )

    def close(self) -> None:
        return None
