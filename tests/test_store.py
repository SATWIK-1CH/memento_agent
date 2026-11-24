from memento_agent.types import (
    FetchStatus,
    PlannedQuery,
    ResearchAnswer,
    SourceRecord,
    SourceType,
)


def test_persist_and_reload_run(store) -> None:
    store.start_run("abc123", "Agonda hotels?", [SourceType.BLOG, SourceType.REDDIT], model="gemini-2.5-flash", search_model="gemini-2.5-flash")
    store.add_sub_queries(
        "abc123",
        0,
        [PlannedQuery(query="Agonda hotels", source_type=SourceType.BLOG, rationale="blogs")],
    )
    store.add_source(
        "abc123",
        SourceRecord(
            id="S1",
            url="https://travelblog.example/agonda",
            title="Guide",
            source_type=SourceType.BLOG,
            fetch_status=FetchStatus.OK,
            excerpt="Simrose",
        ),
    )
    answer = ResearchAnswer(summary="Go to Agonda.", sections=[], limitations=[])
    store.complete_run("abc123", answer=answer, markdown="# hi\n", iterations=1, searches_used=2)

    row = store.get_run("abc123")
    assert row is not None
    assert row["status"] == "completed"
    assert row["answer_markdown"] == "# hi\n"
    assert store.get_sources("abc123")[0]["id"] == "S1"
    assert store.list_runs()[0]["id"] == "abc123"


def test_fail_run(store) -> None:
    store.start_run("x", "q", [SourceType.WEB], model="m", search_model="m")
    store.fail_run("x", "quota")
    assert store.get_run("x")["status"] == "failed"
    assert store.get_run("x")["error"] == "quota"
