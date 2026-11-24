"""Watch or transcribe the most relevant YouTube hits before corroboration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from memento_agent.errors import LLMError, QuotaExhaustedError, RateLimitError
from memento_agent.events import EventBus
from memento_agent.sources import canonical_youtube_url, youtube_video_id
from memento_agent.types import FetchStatus, SourceRecord, SourceType, VideoWatchNotes

TranscriptFetcher = Callable[[str], str]


class YouTubeAnalyzer:
    """Cap Gemini video watches per run; fall back to captions, then title-only."""

    def __init__(
        self,
        llm: Any,
        *,
        max_videos: int = 2,
        excerpt_chars: int = 8_000,
        transcript_fetcher: TranscriptFetcher | None = None,
    ) -> None:
        self._llm = llm
        self._max_videos = max(0, max_videos)
        self._excerpt_chars = excerpt_chars
        self._transcript_fetcher = transcript_fetcher or fetch_youtube_transcript
        self._processed = 0

    def enrich(
        self,
        sources: list[SourceRecord],
        question: str,
        *,
        events: EventBus | None = None,
    ) -> list[SourceRecord]:
        if self._max_videos <= 0 or self._processed >= self._max_videos:
            return sources
        remaining = self._max_videos - self._processed
        candidates = [
            s
            for s in sources
            if s.source_type is SourceType.YOUTUBE and s.fetch_status is FetchStatus.YOUTUBE_META
        ]
        ranked = sorted(candidates, key=lambda s: _relevance(s, question), reverse=True)
        for source in ranked[:remaining]:
            self._enrich_one(source, events=events)
            self._processed += 1
        return sources

    def _enrich_one(self, source: SourceRecord, *, events: EventBus | None) -> None:
        url = canonical_youtube_url(source.display_url or source.url)
        video_id = youtube_video_id(url)
        if events:
            events.emit("watch", f"{source.id} watching {source.title or url}")
        if hasattr(self._llm, "watch_video"):
            try:
                notes = self._llm.watch_video(url, title=source.title)
                source.fetch_status = FetchStatus.YOUTUBE_WATCHED
                source.excerpt = _clip(_format_watch_notes(notes), self._excerpt_chars)
                source.fetch_note = (
                    "Watched with Gemini video understanding; only what was seen/said is used."
                )
                return
            except Exception as exc:
                if events:
                    events.emit(
                        "watch",
                        f"{source.id} video watch failed ({_short(exc)}); trying captions",
                    )

        if not video_id:
            return
        try:
            transcript = self._transcript_fetcher(video_id)
        except Exception as exc:
            if events:
                events.emit("watch", f"{source.id} captions unavailable ({_short(exc)}); title only")
            return
        if not (transcript or "").strip():
            return
        source.fetch_status = FetchStatus.YOUTUBE_TRANSCRIPT
        source.excerpt = _clip(
            "TRANSCRIPT (captions only; video was not watched)\n\n" + transcript.strip(),
            self._excerpt_chars,
        )
        source.fetch_note = (
            "Gemini video watch failed or was skipped; used YouTube captions only. "
            "Auto-captions can be messy — treat details as unconfirmed unless another source agrees."
        )


def fetch_youtube_transcript(video_id: str) -> str:
    """Captions via youtube-transcript-api (no key). Prefer English, then Hindi."""
    from youtube_transcript_api import YouTubeTranscriptApi

    languages = ["en", "en-US", "en-GB", "hi"]
    snippets: Any
    api = YouTubeTranscriptApi()
    if hasattr(api, "fetch"):
        snippets = api.fetch(video_id, languages=languages)
    else:
        snippets = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)  # type: ignore[attr-defined]
    return _format_transcript(snippets)


def _format_transcript(snippets: Any) -> str:
    lines: list[str] = []
    iterable = snippets
    if hasattr(snippets, "to_raw_data"):
        iterable = snippets.to_raw_data()
    for item in iterable or []:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            start = item.get("start")
        else:
            text = str(getattr(item, "text", "") or "").strip()
            start = getattr(item, "start", None)
        if not text:
            continue
        if start is not None:
            try:
                lines.append(f"[{_fmt_ts(float(start))}] {text}")
            except (TypeError, ValueError):
                lines.append(text)
        else:
            lines.append(text)
    return "\n".join(lines)


def _format_watch_notes(notes: VideoWatchNotes) -> str:
    lines = ["WATCHED via Gemini video understanding (only what was seen or said):"]
    if notes.summary:
        lines.append(f"Summary: {notes.summary}")
    if notes.stays_shown:
        lines.append("Stays shown: " + "; ".join(notes.stays_shown))
    if notes.prices_stated:
        lines.append("Prices stated: " + "; ".join(notes.prices_stated))
    if notes.pros:
        lines.append("Pros: " + "; ".join(notes.pros))
    if notes.cons:
        lines.append("Cons: " + "; ".join(notes.cons))
    if notes.timestamps:
        lines.append("Timestamps: " + "; ".join(notes.timestamps))
    if notes.caveats:
        lines.append("Caveats: " + "; ".join(notes.caveats))
    return "\n".join(lines)


def _relevance(source: SourceRecord, question: str) -> int:
    hay = f"{source.title} {source.snippet} {source.excerpt}".lower()
    tokens = {t for t in _tokenize(question) if len(t) > 3}
    if not tokens:
        return 0
    return sum(1 for t in tokens if t in hay)


def _tokenize(text: str) -> list[str]:
    return [p for p in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if p]


def _fmt_ts(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _short(exc: BaseException) -> str:
    if isinstance(exc, (LLMError, RateLimitError, QuotaExhaustedError)):
        return type(exc).__name__
    return type(exc).__name__
