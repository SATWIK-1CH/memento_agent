from __future__ import annotations

import httpx

from memento_agent.fetch import Fetcher, looks_parked
from memento_agent.types import FetchStatus, SearchHit, SourceType


def _client(handler) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, follow_redirects=True)


def test_looks_parked_detects_for_sale_page() -> None:
    html = "<html><head><title>This domain is for sale</title></head><body>Buy this domain at HugeDomains</body></html>"
    assert looks_parked(html, title="This domain is for sale") is True
    assert looks_parked("<html><title>Simrose Beach Resort</title><body>Book a hut</body></html>", title="Simrose") is False


def test_html_fetch_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><head><title>Agonda guide</title></head><body><article><p>Simrose is a beach hotel.</p></article></body></html>",
        )

    fetcher = Fetcher(client=_client(handler), delay_s=0)
    hit = SearchHit(url="https://travelblog.example/agonda", title="", source_type=SourceType.BLOG)
    rec = fetcher.fetch(hit, source_id="S1", iteration=1)
    assert rec.fetch_status in {FetchStatus.OK, FetchStatus.SNIPPET_ONLY}
    assert rec.id == "S1"


def test_406_retries_with_broader_accept() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("accept", ""))
        if "*/*" not in request.headers.get("accept", "") or request.headers.get("accept") != "*/*":
            if request.headers.get("accept") == "*/*":
                return httpx.Response(200, text="<html><title>Hotel</title><body><p>Rooms on the beach at Simrose Agonda Goa resort stay.</p></body></html>")
            return httpx.Response(406, text="Not Acceptable")
        return httpx.Response(200, text="<html><title>Hotel</title><body><p>Rooms on the beach at Simrose Agonda Goa resort stay.</p></body></html>")

    fetcher = Fetcher(client=_client(handler), delay_s=0)
    rec = fetcher.fetch(
        SearchHit(url="https://hotel.example/", title="Hotel", source_type=SourceType.WEB),
        source_id="S1",
        iteration=1,
    )
    assert any(a == "*/*" for a in seen)
    assert rec.fetch_status in {FetchStatus.OK, FetchStatus.SNIPPET_ONLY, FetchStatus.FAILED}


def test_parked_domain_is_dead_not_citable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="<html><head><title>This domain is for sale</title></head><body>This domain is parked. Buy this domain today.</body></html>",
        )

    fetcher = Fetcher(client=_client(handler), delay_s=0)
    rec = fetcher.fetch(
        SearchHit(url="https://dead-hotel.example/", title="Simrose official", source_type=SourceType.WEB),
        source_id="S1",
        iteration=1,
    )
    assert rec.fetch_status is FetchStatus.DEAD
    assert rec.citable is False


def test_reddit_falls_back_to_snippet_on_403() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="blocked")

    fetcher = Fetcher(client=_client(handler), delay_s=0)
    rec = fetcher.fetch(
        SearchHit(
            url="https://www.reddit.com/r/goatravel/comments/abc/agonda/",
            title="Agonda?",
            snippet="Stay at Simrose, skip the party strip.",
            source_type=SourceType.REDDIT,
        ),
        source_id="S2",
        iteration=1,
    )
    assert rec.fetch_status in {FetchStatus.BLOCKED, FetchStatus.SNIPPET_ONLY}
    assert "Simrose" in rec.excerpt
    assert "snippet" in rec.fetch_note.lower() or "blocked" in rec.fetch_note.lower() or "could not open" in rec.fetch_note.lower()


def test_youtube_uses_oembed_title_only() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "oembed" in str(request.url):
            return httpx.Response(
                200,
                json={"title": "Agonda Beach Goa 2024", "author_name": "TravelChan"},
            )
        return httpx.Response(200, text='<html><meta property="og:description" content="A walk along Agonda."><title>watch</title></html>')

    fetcher = Fetcher(client=_client(handler), delay_s=0)
    rec = fetcher.fetch(
        SearchHit(
            url="https://www.youtube.com/watch?v=dQw4w9wg",
            title="",
            snippet="hotel tour",
            source_type=SourceType.YOUTUBE,
        ),
        source_id="S3",
        iteration=1,
    )
    assert rec.fetch_status is FetchStatus.YOUTUBE_META
    assert rec.title == "Agonda Beach Goa 2024"
    assert "not watched" in rec.fetch_note.lower() or "featured" in rec.fetch_note.lower()
