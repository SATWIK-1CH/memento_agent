from memento_agent.corroborate import merge_followups
from memento_agent.types import (
    EntityCheck,
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
