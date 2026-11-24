"""Watch or transcribe the most relevant YouTube hits before corroboration."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from memento_agent.errors import LLMError, QuotaExhaustedError, RateLimitError
from memento_agent.events import EventBus
from memento_agent.sources import canonical_youtube_url, youtube_video_id
from memento_agent.types import FetchStatus, SourceRecord, SourceType, VideoWatchNotes

TranscriptFetcher = Callable[[str], str]

_STOP = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "your",
    "are",
    "was",
    "were",
    "best",
    "which",
    "other",
    "search",
    "cover",
    "trip",
    "areas",
    "area",
}
_SHORT_KEEP = {"goa", "hi", "in"}
_GENERIC_QUERY = {"hotels", "hotel", "best", "youtube", "search", "travel", "blogs", "reddit"}
_GENERIC_TITLE = (
    "top 10",
    "top10",
    "10 best",
    "best hotels in",
    "must visit",
    "things to do",
)
_INDIAN_LANGS = ("en", "en-US", "en-GB", "hi", "hi-IN", "mr", "kn", "te", "ta", "ml", "bn", "gu", "pa", "or", "as")


class YouTubeAnalyzer:
    """Cap Gemini video watches per run; fall back to captions, then title-only."""

    def __init__(
        self,
        llm: Any,
        *,
        max_videos: int = 2,
        excerpt_chars: int = 24_000,
        transcript_fetcher: TranscriptFetcher | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        watch_gap_s: float = 1.5,
    ) -> None:
        self._llm = llm
        self._max_videos = max(0, max_videos)
        self._excerpt_chars = excerpt_chars
        self._transcript_fetcher = transcript_fetcher or fetch_youtube_transcript
        self._sleeper = sleeper
        self._watch_gap_s = watch_gap_s
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
        ranked = sorted(
            candidates,
            key=lambda s: _relevance(s, question),
            reverse=True,
        )
        for index, source in enumerate(ranked[:remaining]):
            if index and self._watch_gap_s:
                self._sleeper(self._watch_gap_s)
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
                source.excerpt = _clip_keep_prices(_format_watch_notes(notes), self._excerpt_chars)
                source.fetch_note = (
                    "Watched with Gemini video understanding; only what was seen/said is used."
                )
                if events:
                    events.emit("watch", f"{source.id} watched")
                return
            except Exception as exc:
                if events:
                    events.emit(
                        "watch",
                        f"{source.id} video watch failed ({_short(exc)}); trying captions",
                    )

        if not video_id:
            if events:
                events.emit("watch", f"{source.id} failed; title only")
            return
        try:
            transcript = self._transcript_fetcher(video_id)
        except Exception as exc:
            if events:
                events.emit("watch", f"{source.id} captions unavailable ({_short(exc)}); title only")
            return
        if not (transcript or "").strip():
            if events:
                events.emit("watch", f"{source.id} failed; title only")
            return
        source.fetch_status = FetchStatus.YOUTUBE_TRANSCRIPT
        source.excerpt = _clip_keep_prices(
            "TRANSCRIPT (captions only; video was not watched)\n\n" + transcript.strip(),
            self._excerpt_chars,
        )
        source.fetch_note = (
            "Gemini video watch failed or was skipped; used YouTube captions only. "
            "Auto-captions can be messy — treat details as unconfirmed unless another source agrees."
        )
        if events:
            events.emit("watch", f"{source.id} captions")


def fetch_youtube_transcript(video_id: str) -> str:
    """Captions via youtube-transcript-api (no key). English, then Indian languages."""
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    snippets = _fetch_preferred_track(api, video_id)
    return _format_transcript(snippets)


def _fetch_preferred_track(api: Any, video_id: str) -> Any:
    listed = None
    if hasattr(api, "list"):
        try:
            listed = api.list(video_id)
        except Exception:
            listed = None
    if listed is not None:
        snippets = _pick_from_list(listed)
        if snippets is not None:
            return snippets
    if hasattr(api, "fetch"):
        return api.fetch(video_id, languages=list(_INDIAN_LANGS))
    getter = getattr(type(api), "get_transcript", None)
    if getter is not None:
        return getter(video_id, languages=list(_INDIAN_LANGS))
    raise LLMError(f"No transcript method available for video {video_id}")


def _pick_from_list(listed: Any) -> Any | None:
    finders = []
    if hasattr(listed, "find_manually_created_transcript"):
        finders.append(listed.find_manually_created_transcript)
    if hasattr(listed, "find_generated_transcript"):
        finders.append(listed.find_generated_transcript)
    if hasattr(listed, "find_transcript"):
        finders.append(listed.find_transcript)
    for finder in finders:
        try:
            track = finder(list(_INDIAN_LANGS))
        except Exception:
            continue
        if track is None:
            continue
        if hasattr(track, "fetch"):
            return track.fetch()
        return track
    return None


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
    title = (source.title or "").lower()
    tokens = [
        t
        for t in _tokenize(question)
        if t not in _STOP and (len(t) >= 3 or t in _SHORT_KEEP)
    ]
    if not tokens:
        return 0
    distinctive = [t for t in tokens if t not in _GENERIC_QUERY]
    score = 0
    for token in tokens:
        if token in hay:
            score += 3 if token in title else 1
            if len(token) >= 5:
                score += 1
    for token in distinctive:
        if token in title:
            score += 6
        elif token in hay:
            score += 2
    if any(g in title for g in _GENERIC_TITLE):
        score -= 10
        if not any(t in title for t in distinctive):
            score -= 6
    return score


def _tokenize(text: str) -> list[str]:
    return [p for p in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if p]


def _fmt_ts(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _clip_keep_prices(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    kept = text[:head]
    rest = text[head:]
    price_lines = [
        line
        for line in rest.splitlines()
        if any(mark in line.lower() for mark in ("₹", "rs.", "rs ", "inr", "rupee", "price", "/night", "per night"))
    ]
    extra = "\n".join(price_lines)
    budget = limit - len(kept) - 20
    if extra and budget > 40:
        kept = kept.rstrip() + "\n…\n" + extra[:budget]
    else:
        kept = text[: limit - 1].rstrip() + "…"
    return kept[:limit]


def _short(exc: BaseException) -> str:
    msg = str(exc).strip().split("\n", 1)[0][:220]
    name = type(exc).__name__
    if isinstance(exc, (LLMError, RateLimitError, QuotaExhaustedError)) or msg:
        return f"{name}: {msg}" if msg else name
    return name
