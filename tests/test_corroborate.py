from memento_agent.corroborate import enforce_coverage, limitations_from_sources, merge_followups
from memento_agent.types import (
    EntityCheck,
    ExtractedEntity,
    FetchStatus,
    FollowupQuery,
    Judgment,
    SourceRecord,
    SourceType,
)


def test_uncorroborated_entity_adds_official_and_reddit_queries() -> None:
    judgment = Judgment(
        enough=False,
        followup_queries=[
            FollowupQuery(query="Agonda Palolem", source_type=SourceType.BLOG, gap="areas")
        ],
        entity_checks=[
            EntityCheck(
                name="Drop Inn Cafe",
                kind="hotel",
                corroborated=False,
                source_ids=["S1"],
                source_types=[SourceType.BLOG],
                notes="Only a listicle; might be a restaurant.",
            )
        ],
    )
    sources = [
        SourceRecord(
            id="S1",
            url="https://listicle.example/agonda",
            source_type=SourceType.BLOG,
            fetch_status=FetchStatus.OK,
            excerpt="Drop Inn Cafe hotel",
        )
    ]
    pending = merge_followups(judgment, sources)
    queries = [p.query.lower() for p in pending]
    assert any("palolem" in q for q in queries)
    assert any("official site" in q for q in queries)
    assert any("reddit.com" in q for q in queries)


def test_enforce_coverage_blocks_early_enough() -> None:
    judgment = Judgment(enough=True, gaps=[], missing_source_types=[SourceType.REDDIT], entity_checks=[])
    entities = [
        ExtractedEntity(
            name="Cuba Agonda",
            kind="hotel",
            source_ids=["S1"],
            source_types=[SourceType.YOUTUBE],
            confidence="high",
        )
    ]
    sources = [
        SourceRecord(
            id="S1",
            url="https://www.youtube.com/watch?v=x",
            source_type=SourceType.YOUTUBE,
            fetch_status=FetchStatus.YOUTUBE_META,
            title="Cuba Agonda",
            excerpt="Cuba Agonda pool",
        )
    ]
    out = enforce_coverage(
        judgment,
        entities,
        sources,
        [SourceType.YOUTUBE, SourceType.BLOG, SourceType.REDDIT],
        iteration=1,
        max_iterations=3,
    )
    assert out.enough is False
    assert any("cuba" in g.lower() for g in out.gaps)


def test_missing_source_followup_uses_question_topic() -> None:
    judgment = Judgment(enough=True, gaps=[], missing_source_types=[SourceType.YOUTUBE], entity_checks=[])
    out = enforce_coverage(
        judgment,
        [],
        [
            SourceRecord(
                id="S1",
                url="https://docs.python.org/3/whatsnew/3.13.html",
                source_type=SourceType.WEB,
                fetch_status=FetchStatus.OK,
                excerpt="Python 3.13 release notes.",
            )
        ],
        [SourceType.WEB, SourceType.YOUTUBE],
        iteration=1,
        max_iterations=3,
        question="Python 3.13 release notes. Search YouTube and the web.",
    )
    queries = [f.query.lower() for f in out.followup_queries]
    assert queries
    assert not any("agonda" in q for q in queries)
    assert any("python 3.13" in f.query.lower() and f.source_type is SourceType.YOUTUBE for f in out.followup_queries)


def test_limitations_collapse_reddit_403s() -> None:
    sources = [
        SourceRecord(
            id=f"S{i}",
            url=f"https://www.reddit.com/r/goatravel/comments/{i}/",
            source_type=SourceType.REDDIT,
            fetch_status=FetchStatus.BLOCKED,
            fetch_note=f"HTTP 403 from old.reddit.com thread {i}",
            excerpt="snippet",
        )
        for i in range(8)
    ]
    notes = limitations_from_sources(sources)
    assert notes == ["8 Reddit threads blocked; used search snippets."]
