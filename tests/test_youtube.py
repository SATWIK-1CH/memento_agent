from memento_agent.types import FetchStatus, SourceRecord, SourceType, VideoWatchNotes
from memento_agent.youtube import YouTubeAnalyzer, _format_transcript


class FakeWatchLLM:
    def __init__(self, notes: VideoWatchNotes | None = None, error: Exception | None = None) -> None:
        self.notes = notes or VideoWatchNotes(
            summary="Walk along Agonda beach.",
            stays_shown=["Simrose beach huts"],
            prices_stated=["₹4500 a night spoken at 2:10"],
            pros=["quiet at the north end"],
            cons=["loud music near the south shacks"],
            timestamps=["2:10 Simrose price"],
        )
        self.error = error
        self.calls: list[str] = []

    def watch_video(self, url: str, *, title: str = "") -> VideoWatchNotes:
        self.calls.append(url)
        if self.error:
            raise self.error
        return self.notes


def _yt(source_id: str, *, title: str, snippet: str = "") -> SourceRecord:
    return SourceRecord(
        id=source_id,
        url=f"https://www.youtube.com/watch?v={source_id.lower()}",
        title=title,
        snippet=snippet,
        source_type=SourceType.YOUTUBE,
        fetch_status=FetchStatus.YOUTUBE_META,
        fetch_note="title only",
        excerpt=snippet,
    )


def test_watch_success_marks_watched() -> None:
    llm = FakeWatchLLM()
    analyzer = YouTubeAnalyzer(llm, max_videos=2, transcript_fetcher=lambda _vid: "")
    sources = [_yt("S1", title="Agonda beach hotels 2024", snippet="Simrose huts Agonda Goa")]
    analyzer.enrich(sources, "Best hotels in Agonda Goa")
    assert sources[0].fetch_status is FetchStatus.YOUTUBE_WATCHED
    assert "Simrose" in sources[0].excerpt
    assert "WATCHED" in sources[0].excerpt
    assert llm.calls


def test_watch_failure_falls_back_to_transcript() -> None:
    llm = FakeWatchLLM(error=RuntimeError("503 high demand"))
    analyzer = YouTubeAnalyzer(
        llm,
        max_videos=2,
        transcript_fetcher=lambda vid: "[{'text': 'x'}]" and "We stayed at Simrose. It was 4000 a night.",
    )
    sources = [_yt("S1", title="Agonda", snippet="hotels")]
    analyzer.enrich(sources, "Agonda hotels")
    assert sources[0].fetch_status is FetchStatus.YOUTUBE_TRANSCRIPT
    assert "captions" in sources[0].fetch_note.lower()
    assert "Simrose" in sources[0].excerpt


def test_transcript_failure_stays_title_only() -> None:
    llm = FakeWatchLLM(error=RuntimeError("no video"))

    def boom(_vid: str) -> str:
        raise RuntimeError("no captions")

    analyzer = YouTubeAnalyzer(llm, max_videos=2, transcript_fetcher=boom)
    sources = [_yt("S1", title="Agonda vlog")]
    analyzer.enrich(sources, "Agonda")
    assert sources[0].fetch_status is FetchStatus.YOUTUBE_META


def test_cap_skips_extra_videos() -> None:
    llm = FakeWatchLLM()
    analyzer = YouTubeAnalyzer(llm, max_videos=1, transcript_fetcher=lambda _vid: "")
    sources = [
        _yt("AAA", title="Agonda hotels tour Simrose Palolem", snippet="Agonda hotels Simrose"),
        _yt("BBB", title="Unrelated cooking", snippet="pasta"),
    ]
    analyzer.enrich(sources, "Best hotels in Agonda Goa Simrose")
    statuses = {s.id: s.fetch_status for s in sources}
    watched = [s for s in sources if s.fetch_status is FetchStatus.YOUTUBE_WATCHED]
    assert len(watched) == 1
    assert watched[0].id == "AAA"
    assert statuses["BBB"] is FetchStatus.YOUTUBE_META
    assert len(llm.calls) == 1


def test_relevance_keeps_goa_and_penalises_generic_listicle() -> None:
    from memento_agent.youtube import _relevance

    specific = _yt("AAA", title="Simrose Agonda review", snippet="goa hut stay")
    generic = _yt("BBB", title="Top 10 best hotels in India", snippet="listicle")
    question = "Best hotels in Agonda Goa Simrose"
    assert _relevance(specific, question) > _relevance(generic, question)


def test_short_logs_exception_message() -> None:
    from memento_agent.errors import RateLimitError
    from memento_agent.youtube import _short

    text = _short(RateLimitError("Gemini rate-limited this request (HTTP 429)."))
    assert "RateLimitError" in text
    assert "429" in text


def test_format_transcript_timestamps() -> None:
    text = _format_transcript(
        [{"text": "Hello Agonda", "start": 75.2, "duration": 2.0}, {"text": "Simrose", "start": 80}]
    )
    assert "[1:15] Hello Agonda" in text
    assert "[1:20] Simrose" in text


def test_fetch_youtube_transcript_uses_instance_api(monkeypatch) -> None:
    from memento_agent.youtube import fetch_youtube_transcript

    class Wrapper:
        def fetch(self, video_id, languages=None):
            assert video_id == "abc"
            assert "en" in (languages or [])
            assert "hi" in (languages or [])
            return [{"text": "hello", "start": 9}]

    monkeypatch.setattr("youtube_transcript_api.YouTubeTranscriptApi", Wrapper)
    text = fetch_youtube_transcript("abc")
    assert "[0:09] hello" in text


def test_fetch_youtube_transcript_accepts_hindi_listed_track(monkeypatch) -> None:
    from memento_agent.youtube import fetch_youtube_transcript

    class Track:
        def fetch(self):
            return [{"text": "Agonda Hindi vlog", "start": 12}]

    class Listed:
        def find_manually_created_transcript(self, languages):
            raise RuntimeError("no manual en")

        def find_generated_transcript(self, languages):
            raise RuntimeError("no generated en")

        def find_transcript(self, languages):
            assert "hi" in languages
            return Track()

    class Wrapper:
        def list(self, video_id):
            assert video_id == "hindi1"
            return Listed()

    monkeypatch.setattr("youtube_transcript_api.YouTubeTranscriptApi", Wrapper)
    text = fetch_youtube_transcript("hindi1")
    assert "Agonda Hindi vlog" in text


def test_format_transcript_accepts_objects() -> None:
    class Snippet:
        def __init__(self) -> None:
            self.text = "hi"
            self.start = 3

    assert "[0:03] hi" in _format_transcript([Snippet()])
