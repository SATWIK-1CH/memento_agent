"""Source-type detection from questions and URLs."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from memento_agent.types import SourceType

_TYPE_PATTERNS: list[tuple[SourceType, re.Pattern[str]]] = [
    (SourceType.YOUTUBE, re.compile(r"\byoutube\b|\byou[\s-]?tube\b|\byt videos?\b", re.I)),
    (SourceType.REDDIT, re.compile(r"\breddit\b|\bredditors\b|\bsubreddit\b", re.I)),
    (
        SourceType.BLOG,
        re.compile(r"\bblogs?\b|\btravel blogs?\b|\bnews\b|\barticles?\b", re.I),
    ),
    (SourceType.WEB, re.compile(r"\bgeneral web\b|\bthe web\b|\bgoogle\b|\bweb search\b", re.I)),
]

# Hosts that are clearly blogs, magazines, or travel/news publications.
_BLOG_HOST_MARKERS = (
    "medium.com",
    "substack.com",
    "blogspot",
    "wordpress",
    "tripadvisor",
    "lonelyplanet",
    "cntraveler",
    "timeout.com",
    "theguardian",
    "nytimes",
    "bbc.",
    "cnn.com",
    "washingtonpost",
    "travelandleisure",
    "conde nast",
    "condenasstraveler",
    "nomadicmatt",
    "thepoortraveler",
    "backpacker",
    "wanderlust",
    "booking.com",
    "expedia",
    "hotels.com",
    "makemytrip",
    "cleartrip",
)

_ALL_TYPES = (SourceType.YOUTUBE, SourceType.REDDIT, SourceType.BLOG, SourceType.WEB)


def detect_source_types(question: str) -> list[SourceType]:
    """Return source types the user asked for; default to all four if unspecified."""
    found: list[SourceType] = []
    for source_type, pattern in _TYPE_PATTERNS:
        if pattern.search(question) and source_type not in found:
            found.append(source_type)
    return found or list(_ALL_TYPES)


def youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")
    parts = [p for p in path.split("/") if p]
    if "youtu.be" in host:
        return parts[0] if parts else None
    for marker in ("shorts", "embed", "live"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    query = parse_qs(parsed.query)
    vid = (query.get("v") or [None])[0]
    return vid or None


def canonical_youtube_url(url: str) -> str:
    video_id = youtube_video_id(url)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return url


def is_grounding_redirect(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host == "vertexaisearch.cloud.google.com" or host.endswith(".vertexaisearch.cloud.google.com")


def classify_url(url: str, *, hinted: SourceType | None = None) -> SourceType:
    """Classify a result URL. A search hint wins only when the host is ambiguous."""
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if "youtube.com" in host or host.endswith("youtu.be") or host == "youtu.be":
        return SourceType.YOUTUBE
    if "reddit.com" in host or host.endswith("redd.it") or host == "redd.it":
        return SourceType.REDDIT
    if any(marker in host for marker in _BLOG_HOST_MARKERS):
        return SourceType.BLOG
    if hinted is SourceType.BLOG:
        return SourceType.BLOG
    return hinted or SourceType.WEB


def steer_query(query: str, source_type: SourceType) -> str:
    """Add site: / keyword steering so one search backend can target a source type."""
    q = query.strip()
    if source_type is SourceType.YOUTUBE:
        if "site:youtube.com" not in q.lower():
            q = f"{q} site:youtube.com"
    elif source_type is SourceType.REDDIT:
        if "site:reddit.com" not in q.lower():
            q = f"{q} site:reddit.com"
    elif source_type is SourceType.BLOG:
        if "site:" not in q.lower():
            q = f'{q} (blog OR review OR "travel guide") -site:youtube.com -site:reddit.com'
    return q
