"""Shared dataclasses and Pydantic schemas used across the agent loop."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    YOUTUBE = "youtube"
    REDDIT = "reddit"
    BLOG = "blog"
    WEB = "web"

    @property
    def label(self) -> str:
        return {
            SourceType.YOUTUBE: "YouTube",
            SourceType.REDDIT: "Reddit",
            SourceType.BLOG: "blog/news",
            SourceType.WEB: "web",
        }[self]


class FetchStatus(str, Enum):
    OK = "ok"
    YOUTUBE_META = "youtube_meta"
    BLOCKED = "blocked"
    FAILED = "failed"
    SNIPPET_ONLY = "snippet_only"
    DEAD = "dead"


class SearchHit(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    source_type: SourceType = SourceType.WEB
    query: str = ""


class PlannedQuery(BaseModel):
    query: str
    source_type: SourceType
    rationale: str = ""


class ResearchPlan(BaseModel):
    restated_question: str = ""
    source_types: list[SourceType] = Field(default_factory=list)
    sub_queries: list[PlannedQuery] = Field(default_factory=list)


class FollowupQuery(BaseModel):
    query: str
    source_type: SourceType
    gap: str = ""


class EntityCheck(BaseModel):
    """Cross-source check for a named place, hotel, or similar entity."""

    name: str
    kind: str = "place"
    corroborated: bool = False
    source_ids: list[str] = Field(default_factory=list)
    source_types: list[SourceType] = Field(default_factory=list)
    official_url: str | None = None
    notes: str = ""


class Judgment(BaseModel):
    enough: bool
    rationale: str = ""
    corroborated_points: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    missing_source_types: list[SourceType] = Field(default_factory=list)
    entity_checks: list[EntityCheck] = Field(default_factory=list)
    followup_queries: list[FollowupQuery] = Field(default_factory=list)


class RankedItem(BaseModel):
    name: str
    why: str
    tier: str | None = None
    source_ids: list[str] = Field(default_factory=list)


class AnswerSection(BaseModel):
    title: str
    narrative: str = ""
    items: list[RankedItem] = Field(default_factory=list)


class ResearchAnswer(BaseModel):
    summary: str
    sections: list[AnswerSection] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class SourceRecord(BaseModel):
    id: str
    url: str
    final_url: str = ""
    title: str = ""
    source_type: SourceType = SourceType.WEB
    snippet: str = ""
    fetch_status: FetchStatus = FetchStatus.SNIPPET_ONLY
    fetch_note: str = ""
    excerpt: str = ""
    query: str = ""
    iteration: int = 0

    @property
    def display_url(self) -> str:
        return self.final_url or self.url

    @property
    def evidence_text(self) -> str:
        return (self.excerpt or self.snippet or "").strip()

    @property
    def citable(self) -> bool:
        """True only when there is something a writer may actually quote.

        Dead/parked URLs are never cited. Failed or blocked fetches are cited
        only if the search snippet (or extracted excerpt) is non-empty.
        YouTube title/description counts as evidence even without a page body.
        """
        if self.fetch_status is FetchStatus.DEAD:
            return False
        if self.fetch_status is FetchStatus.YOUTUBE_META:
            return bool((self.title or self.evidence_text).strip())
        return bool(self.evidence_text)


class ResearchResult(BaseModel):
    run_id: str
    question: str
    source_types: list[SourceType]
    answer: ResearchAnswer
    markdown: str
    sources: list[SourceRecord]
    sub_queries: list[PlannedQuery] = Field(default_factory=list)
    iterations: int = 0
    searches_used: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
