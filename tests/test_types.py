from memento_agent.types import FetchStatus, SourceRecord, SourceType


def test_failed_empty_source_is_not_citable() -> None:
    rec = SourceRecord(
        id="S1",
        url="https://example.com/x",
        source_type=SourceType.WEB,
        fetch_status=FetchStatus.FAILED,
        excerpt="",
        snippet="",
    )
    assert rec.citable is False


def test_blocked_with_snippet_is_citable() -> None:
    rec = SourceRecord(
        id="S2",
        url="https://www.reddit.com/r/goatravel/comments/x/",
        source_type=SourceType.REDDIT,
        fetch_status=FetchStatus.BLOCKED,
        snippet="Stay at Simrose",
        excerpt="Stay at Simrose",
        fetch_note="HTTP 403",
    )
    assert rec.citable is True


def test_youtube_title_only_is_citable() -> None:
    rec = SourceRecord(
        id="S3",
        url="https://www.youtube.com/watch?v=aaa",
        title="Agonda Beach vlog",
        source_type=SourceType.YOUTUBE,
        fetch_status=FetchStatus.YOUTUBE_META,
        excerpt="",
        snippet="",
    )
    assert rec.citable is True


def test_dead_is_never_citable() -> None:
    rec = SourceRecord(
        id="S4",
        url="https://parked.example/",
        source_type=SourceType.WEB,
        fetch_status=FetchStatus.DEAD,
        excerpt="Buy this domain",
        snippet="Buy this domain",
    )
    assert rec.citable is False
