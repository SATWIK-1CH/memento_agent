from memento_agent.entities import (
    extract_entities,
    merge_missing_entities,
    scrub_citations,
    source_mentions,
    tidy_limitations,
    tidy_open_questions,
)
from memento_agent.types import (
    AnswerSection,
    FetchStatus,
    RankedItem,
    ResearchAnswer,
    SourceRecord,
    SourceType,
)


def test_extracts_stays_and_prices_from_booking_style_page() -> None:
    sources = [
        SourceRecord(
            id="S3",
            url="https://www.booking.com/agonda",
            title="Agonda hotels — Booking.com",
            source_type=SourceType.WEB,
            fetch_status=FetchStatus.OK,
            excerpt=(
                "Places to stay: Agonda Villas, The Bay Agonda, Simrose, The Rose. "
                "Simrose from ₹4,500 /night. The Bay Agonda Rs 7000 per night."
            ),
        ),
        SourceRecord(
            id="S8",
            url="https://blog.example/first-timers-agonda",
            title="Agonda first-timers guide",
            source_type=SourceType.BLOG,
            fetch_status=FetchStatus.OK,
            excerpt="We liked Jardim A Mar and O2H. Palolem and Patnem are easy day trips. Cola is quieter.",
        ),
        SourceRecord(
            id="S4",
            url="https://www.tripadvisor.com/agonda",
            title="Om Sai Agonda",
            source_type=SourceType.BLOG,
            fetch_status=FetchStatus.OK,
            excerpt="Om Sai is a simple stay. 4.2/5. Rooms from INR 2500.",
        ),
        SourceRecord(
            id="S20",
            url="https://www.youtube.com/watch?v=oasis",
            title="Oasis Beach Resort Agonda tour",
            source_type=SourceType.YOUTUBE,
            fetch_status=FetchStatus.YOUTUBE_META,
            excerpt="Channel: TravelChan",
        ),
    ]
    names = {e.name.lower() for e in extract_entities(sources, "Best hotels in Agonda, Goa")}
    for expected in (
        "agonda villas",
        "the bay agonda",
        "simrose",
        "the rose",
        "jardim a mar",
        "o2h",
        "om sai",
        "oasis beach resort",
        "patnem",
        "cola",
    ):
        assert any(expected in n for n in names), names


def test_scrub_drops_mismatched_cuba_citation() -> None:
    sources = [
        SourceRecord(
            id="S2",
            url="https://blog.example/cottages",
            title="Agonda Cottages",
            source_type=SourceType.BLOG,
            fetch_status=FetchStatus.OK,
            excerpt="Agonda Cottages are wood and coconut palm cottages on the sand.",
        ),
        SourceRecord(
            id="S18",
            url="https://www.youtube.com/watch?v=hindi",
            title="Agonda beach vlog",
            source_type=SourceType.YOUTUBE,
            fetch_status=FetchStatus.YOUTUBE_META,
            excerpt="Generic Hindi walk along Agonda beach. No hotel names.",
        ),
        SourceRecord(
            id="S7",
            url="https://blog.example/cuba",
            title="Cuba Agonda review",
            source_type=SourceType.BLOG,
            fetch_status=FetchStatus.OK,
            excerpt="Cuba Agonda has a pool and is set back from the beach.",
        ),
    ]
    assert source_mentions(sources[0], "Cuba Agonda") is False
    assert source_mentions(sources[2], "Cuba Agonda") is True
    answer = ResearchAnswer(
        summary="Stays",
        sections=[
            AnswerSection(
                title="Hotels",
                items=[
                    RankedItem(
                        name="Cuba Agonda",
                        why="wood and coconut palm cottages",
                        source_ids=["S2", "S18", "S7"],
                    )
                ],
            )
        ],
    )
    scrub_citations(answer, sources)
    assert answer.sections[0].items[0].source_ids == ["S7"]
    assert "coconut" not in answer.sections[0].items[0].why.lower()


def test_merge_adds_dropped_hotels() -> None:
    sources = [
        SourceRecord(
            id="S3",
            url="https://www.booking.com/agonda",
            title="Booking",
            source_type=SourceType.WEB,
            fetch_status=FetchStatus.OK,
            excerpt="Agonda Villas and Simrose on the beach.",
        )
    ]
    entities = extract_entities(sources, "Agonda hotels")
    answer = ResearchAnswer(
        summary="thin",
        sections=[AnswerSection(title="Hotels in Agonda", items=[])],
    )
    merge_missing_entities(answer, entities, sources)
    names = [item.name.lower() for item in answer.sections[0].items]
    assert any("simrose" in n for n in names)
    assert any("villas" in n for n in names)


def test_prices_do_not_leak_across_named_stays() -> None:
    from memento_agent.entities import _prices_near

    blob = "Agonda Villas costs ₹9000 per night. Simrose costs ₹4500 per night."
    villas = _prices_near(blob, "Agonda Villas")
    simrose = _prices_near(blob, "Simrose")
    assert any("9000" in p for p in villas)
    assert not any("4500" in p for p in villas)
    assert any("4500" in p for p in simrose)
    assert not any("9000" in p for p in simrose)


def test_does_not_treat_landmarks_as_hotels() -> None:
    sources = [
        SourceRecord(
            id="S1",
            url="https://blog.example/paris",
            title="Paris first-timers",
            source_type=SourceType.BLOG,
            fetch_status=FetchStatus.OK,
            excerpt=(
                "Hotel Ritz is near the Eiffel Tower. Paris France offers great stays. "
                "Book via American Express."
            ),
        )
    ]
    names = {e.name.lower() for e in extract_entities(sources, "Best hotels in Paris")}
    assert not any("eiffel" in n for n in names), names
    assert not any("american express" in n for n in names), names
    assert not any(n == "paris france" for n in names), names
    assert any("ritz" in n for n in names), names


def test_tidy_open_questions_drops_generic() -> None:
    cleaned = tidy_open_questions(
        ["Needs more research", "What is Simrose charging this month?", "Always check latest prices online"]
    )
    assert cleaned == ["What is Simrose charging this month?"]
    assert tidy_limitations(["Needs more research", "8 Reddit threads blocked; used search snippets."]) == [
        "8 Reddit threads blocked; used search snippets."
    ]
