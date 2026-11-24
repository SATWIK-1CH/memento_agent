"""Deterministic corroboration helpers used alongside the LLM critic."""

from __future__ import annotations

from memento_agent.types import (
    EntityCheck,
    FollowupQuery,
    FetchStatus,
    Judgment,
    PlannedQuery,
    SourceRecord,
    SourceType,
)


def merge_followups(judgment: Judgment, sources: list[SourceRecord]) -> list[PlannedQuery]:
    """Turn critic follow-ups plus uncorroborated entities into the next search batch."""
    pending: list[PlannedQuery] = []
    seen: set[str] = set()

    def add(query: str, source_type: SourceType, rationale: str) -> None:
        key = f"{source_type.value}:{query.strip().lower()}"
        if key in seen or not query.strip():
            return
        seen.add(key)
        pending.append(PlannedQuery(query=query.strip(), source_type=source_type, rationale=rationale))

    for follow in judgment.followup_queries:
        add(follow.query, follow.source_type, follow.gap or "critic follow-up")

    for check in judgment.entity_checks:
        if check.corroborated:
            continue
        if len({s.source_type for s in _sources_for(check, sources)}) >= 2:
            continue
        name = check.name.strip()
        if not name:
            continue
        add(
            f'"{name}" official site',
            SourceType.WEB,
            f"Corroborate {name} against an official/current listing (listicles often mislabel or use old names).",
        )
        add(
            f'"{name}" site:reddit.com',
            SourceType.REDDIT,
            f"See if travellers mention {name} by current name.",
        )
    return pending


def limitations_from_sources(sources: list[SourceRecord]) -> list[str]:
    notes: list[str] = []
    for src in sources:
        if src.fetch_status is FetchStatus.OK:
            continue
        label = src.title or src.display_url
        if src.fetch_status is FetchStatus.YOUTUBE_META:
            notes.append(
                f"YouTube: featured in video “{label}” — title/description only, video not watched."
            )
        elif src.fetch_status is FetchStatus.DEAD:
            notes.append(f"Dead/parked URL not cited: {src.display_url}")
        elif src.fetch_note:
            notes.append(f"{label}: {src.fetch_note}")
    return _unique(notes)


def summarize_entity_checks(checks: list[EntityCheck]) -> list[str]:
    lines: list[str] = []
    for check in checks:
        types = ", ".join(t.value for t in check.source_types) or "none"
        status = "corroborated" if check.corroborated else "uncorroborated"
        extra = f" — {check.notes}" if check.notes else ""
        lines.append(f"{check.name} ({check.kind}): {status} via [{types}]{extra}")
    return lines


def _sources_for(check: EntityCheck, sources: list[SourceRecord]) -> list[SourceRecord]:
    wanted = set(check.source_ids)
    return [s for s in sources if s.id in wanted and s.citable]


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
