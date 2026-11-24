from memento_agent.render import render_markdown
from memento_agent.types import (
    AnswerSection,
    FetchStatus,
    RankedItem,
    ResearchAnswer,
    SourceRecord,
    SourceType,
)


def test_render_groups_sources_and_marks_youtube() -> None:
    answer = ResearchAnswer(
        summary="South Goa overview.",
        sections=[
            AnswerSection(
                title="Hotels",
                items=[
                    RankedItem(name="Simrose", tier="mid-range", why="On the sand.", source_ids=["S1", "S3"])
                ],
            )
        ],
        limitations=[],
    )
    sources = [
        SourceRecord(
            id="S1",
            url="https://travelblog.example/a",
            title="Blog guide",
            source_type=SourceType.BLOG,
            fetch_status=FetchStatus.OK,
            excerpt="Simrose",
        ),
        SourceRecord(
            id="S2",
            url="https://www.youtube.com/watch?v=aaa",
            title="Agonda Beach vlog",
            source_type=SourceType.YOUTUBE,
            fetch_status=FetchStatus.YOUTUBE_WATCHED,
            fetch_note="watched",
            excerpt="WATCHED: Simrose at 2:10",
        ),
        SourceRecord(
            id="S3",
            url="https://parked.example/",
            title="Official",
            source_type=SourceType.WEB,
            fetch_status=FetchStatus.DEAD,
            fetch_note="parked",
        ),
    ]
    md = render_markdown("Agonda hotels", answer, sources)
    assert "### YouTube" in md
    assert "watched" in md.lower()
    assert "dead/parked" in md
    assert "[S1]" in md
