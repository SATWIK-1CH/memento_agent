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


def test_booking_card_layout_does_not_steal_previous_price() -> None:
    from memento_agent.entities import _prices_near

    blob = (
        "Om Sai Agonda 8.5 Exceptional 123 reviews Price from ₹4,500 "
        "Abba's Glory Land 8.2 Very good 40 reviews Price from ₹2,500 "
        "Goa Cottages 8.0 reviews Price from ₹7,500 "
        "Agonda Villas 9.1 Wonderful 80 reviews Price from ₹19,405"
    )
    om = _prices_near(blob, "Om Sai Agonda")
    abba = _prices_near(blob, "Abba's Glory Land")
    cottages = _prices_near(blob, "Goa Cottages")
    villas = _prices_near(blob, "Agonda Villas")
    assert any("4,500" in p for p in om)
    assert not any("2,500" in p for p in om)
    assert any("2,500" in p for p in abba)
    assert not any("4,500" in p for p in abba)
    assert not any("7,500" in p for p in abba)
    assert any("7,500" in p for p in cottages)
    assert not any("2,500" in p for p in cottages)
    assert not any("19,405" in p for p in cottages)
    assert any("19,405" in p for p in villas)
    assert not any("7,500" in p for p in villas)

    sources = [
        SourceRecord(
            id="S1",
            url="https://www.booking.com/agonda",
            title="Agonda hotels — Booking.com",
            source_type=SourceType.WEB,
            fetch_status=FetchStatus.OK,
            excerpt=blob,
        )
    ]
    by_name = {e.name.lower(): e for e in extract_entities(sources, "Best hotels in Agonda, Goa")}
    assert any("4,500" in p for p in by_name["om sai agonda"].prices)
    assert any("19,405" in p for p in by_name["agonda villas"].prices)
    assert not any("7,500" in p for p in by_name["agonda villas"].prices)


def test_rejects_tripadvisor_boilerplate_and_north_goa() -> None:
    sources = [
        SourceRecord(
            id="S1",
            url="https://www.tripadvisor.com/agonda",
            title="Agonda hotels",
            source_type=SourceType.BLOG,
            fetch_status=FetchStatus.OK,
            excerpt=(
                "Hard Rock Calangute and Cavala - Candolim are noisy. "
                "February 2012 – Trip type. Retail Merchandise Products. "
                "with a long stretch of sand. Hidden Gems. live music. "
                "Saturday Night Market—a mix of souvenirs. "
                "Places to stay: Simrose, Agonda Villas."
            ),
        )
    ]
    names = {e.name.lower() for e in extract_entities(sources, "Best hotels in Agonda, Goa")}
    joined = " | ".join(names)
    assert "simrose" in names or any("simrose" in n for n in names)
    assert "calangute" not in joined
    assert "candolim" not in joined
    assert "trip type" not in joined
    assert "merchandise" not in joined
    assert "hidden gems" not in joined
    assert "stretch of sand" not in joined


def test_extracts_cuba_cape_and_o2h_suffix() -> None:
    sources = [
        SourceRecord(
            id="S26",
            url="https://www.youtube.com/watch?v=cuba",
            title="Is Cuba Agonda Worth It? | Agonda Beach Resort Review",
            source_type=SourceType.YOUTUBE,
            fetch_status=FetchStatus.YOUTUBE_META,
            excerpt="Channel: TravelChan",
        ),
        SourceRecord(
            id="S3",
            url="https://thatgoangirl.example/agonda",
            title="That Goan Girl",
            source_type=SourceType.BLOG,
            fetch_status=FetchStatus.OK,
            excerpt="The Cape, Cabo de Rama is this romantic resort above the cliffs.",
        ),
        SourceRecord(
            id="S13",
            url="https://blog.example/o2h",
            title="Agonda stays",
            source_type=SourceType.BLOG,
            fetch_status=FetchStatus.OK,
            excerpt="We walked past The O2H Agonda Beach Resort at the north end.",
        ),
    ]
    ents = extract_entities(sources, "Best hotels in Agonda, Goa")
    names = {e.name.lower() for e in ents}
    assert any("cuba" in n for n in names), names
    assert any("cape" in n for n in names), names
    assert any("o2h" in n for n in names), names


def test_watch_notes_keep_price_qualifiers() -> None:
    sources = [
        SourceRecord(
            id="S9",
            url="https://www.youtube.com/watch?v=royal",
            title="Royal Agonda tour",
            source_type=SourceType.YOUTUBE,
            fetch_status=FetchStatus.YOUTUBE_WATCHED,
            excerpt=(
                "WATCHED via Gemini video understanding (only what was seen or said):\n"
                "Stays shown: Royal Agonda\n"
                "Prices stated: INR 8500 for 2 nights; ₹30,000–40,000 package"
            ),
        )
    ]
    ents = extract_entities(sources, "Best hotels in Agonda")
    royal = next(e for e in ents if "royal" in e.name.lower())
    blob = " ".join(royal.prices)
    assert "8500" in blob
    assert "2 nights" in blob.lower() or "for 2" in blob.lower()
    assert "30,000" in blob or "30000" in blob.replace(",", "")


def test_merge_caps_items_and_adds_also_named() -> None:
    sources = [
        SourceRecord(
            id="S1",
            url="https://example.com/list",
            title="stays",
            source_type=SourceType.WEB,
            fetch_status=FetchStatus.OK,
            excerpt="Places to stay: " + ", ".join(f"Stay{i} Hotel" for i in range(40)),
        )
    ]
    entities = extract_entities(sources, "Agonda hotels")
    answer = ResearchAnswer(
        summary="thin",
        sections=[AnswerSection(title="Hotels in Agonda", items=[])],
    )
    merge_missing_entities(answer, entities, sources)
    hotel_section = next(s for s in answer.sections if "hotel" in s.title.lower())
    assert len(hotel_section.items) <= 30
    if len(entities) > 30:
        assert "also named" in hotel_section.narrative.lower()


def test_tidy_open_questions_drops_generic() -> None:
    cleaned = tidy_open_questions(
        ["Needs more research", "What is Simrose charging this month?", "Always check latest prices online"]
    )
    assert cleaned == ["What is Simrose charging this month?"]
    assert tidy_limitations(["Needs more research", "8 Reddit threads blocked; used search snippets."]) == [
        "8 Reddit threads blocked; used search snippets."
    ]
