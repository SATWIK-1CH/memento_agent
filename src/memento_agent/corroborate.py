"""Deterministic corroboration helpers used alongside the LLM critic."""

from __future__ import annotations

import re

from memento_agent.types import (
    EntityCheck,
    ExtractedEntity,
    FollowupQuery,
    FetchStatus,
    Judgment,
    PlannedQuery,
    SourceRecord,
    SourceType,
)

_REDDIT_UNUSABLE = {FetchStatus.BLOCKED, FetchStatus.SNIPPET_ONLY, FetchStatus.FAILED}


def reddit_unreachable(sources: list[SourceRecord]) -> bool:
    """True when Reddit was attempted but every thread stayed blocked/snippet-only."""
    reddit = [s for s in sources if s.source_type is SourceType.REDDIT]
    if not reddit:
        return False
    return all(s.fetch_status in _REDDIT_UNUSABLE for s in reddit)


def enforce_coverage(
    judgment: Judgment,
    entities: list[ExtractedEntity],
    sources: list[SourceRecord],
    requested: list[SourceType],
    *,
    iteration: int,
    max_iterations: int,
    question: str = "",
) -> Judgment:
    """Do not stop while important stays lack two source types or a requested type is missing."""
    blocked_reddit = reddit_unreachable(sources)
    gaps = list(judgment.gaps)
    missing = list(judgment.missing_source_types)
    checks = list(judgment.entity_checks) or _checks_from_entities(entities)

    for source_type in requested:
        hits = [s for s in sources if s.source_type is source_type]
        if source_type is SourceType.REDDIT and blocked_reddit:
            missing = [t for t in missing if t is not SourceType.REDDIT]
            note = "Reddit threads were blocked; search snippets were used. Not retrying Reddit forever."
            if note not in gaps:
                gaps.append(note)
            continue
        if not hits and source_type not in missing:
            missing.append(source_type)
            gaps.append(f"No {source_type.value} sources yet.")

    need_two_types = _need_two_types(requested, blocked_reddit)
    uncorroborated: list[ExtractedEntity] = []
    if need_two_types:
        for entity in entities:
            if entity.kind == "area":
                continue
            types = set(entity.source_types)
            if len(types) < 2:
                uncorroborated.append(entity)
                gap = f"{entity.name} only appears in {', '.join(t.value for t in types) or 'no'} source type(s)."
                if gap not in gaps:
                    gaps.append(gap)
                _ensure_check(checks, entity, corroborated=False)

    last_round = iteration >= max_iterations
    if uncorroborated or (missing and not last_round):
        judgment.enough = False
    elif last_round:
        judgment.enough = True
    judgment.gaps = gaps
    judgment.missing_source_types = missing
    judgment.entity_checks = checks
    if not last_round:
        extra = _followups_for(uncorroborated, missing, blocked_reddit, question=question)
        have = {(f.query.lower(), f.source_type) for f in judgment.followup_queries}
        for follow in extra:
            key = (follow.query.lower(), follow.source_type)
            if key not in have:
                judgment.followup_queries.append(follow)
                have.add(key)
    return judgment


def merge_followups(
    judgment: Judgment,
    sources: list[SourceRecord],
    *,
    entities: list[ExtractedEntity] | None = None,
    blocked_reddit: bool = False,
) -> list[PlannedQuery]:
    """Turn critic follow-ups plus uncorroborated entities into the next search batch."""
    pending: list[PlannedQuery] = []
    seen: set[str] = set()

    def add(query: str, source_type: SourceType, rationale: str) -> None:
        key = f"{source_type.value}:{query.strip().lower()}"
        if key in seen or not query.strip():
            return
        if source_type is SourceType.REDDIT and blocked_reddit:
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
            f'"{name}" review',
            SourceType.BLOG,
            f"Find an independent review of {name}.",
        )
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

    for entity in entities or []:
        if entity.kind == "area" and not entity.corroborated:
            add(
                f"{entity.name} south goa stay",
                SourceType.BLOG,
                f"Cover {entity.name} as an area mentioned in the sources.",
            )
    return pending[:6]


def limitations_from_sources(sources: list[SourceRecord]) -> list[str]:
    notes: list[str] = []
    reddit_blocked = [
        s
        for s in sources
        if s.source_type is SourceType.REDDIT and s.fetch_status in _REDDIT_UNUSABLE
    ]
    if reddit_blocked:
        notes.append(
            f"{len(reddit_blocked)} Reddit threads blocked; used search snippets."
        )
    yt_transcript = [s for s in sources if s.fetch_status is FetchStatus.YOUTUBE_TRANSCRIPT]
    yt_meta = [s for s in sources if s.fetch_status is FetchStatus.YOUTUBE_META]
    yt_watched = [s for s in sources if s.fetch_status is FetchStatus.YOUTUBE_WATCHED]
    if yt_watched:
        notes.append(f"{len(yt_watched)} YouTube video(s) watched with Gemini.")
    if yt_transcript:
        notes.append(
            f"{len(yt_transcript)} YouTube video(s) captions-only (watch failed or was rate-limited)."
        )
    if yt_meta:
        notes.append(
            f"{len(yt_meta)} YouTube video(s) title/description only; not watched."
        )
    for src in sources:
        if src.source_type is SourceType.REDDIT:
            continue
        if src.fetch_status in {
            FetchStatus.OK,
            FetchStatus.YOUTUBE_WATCHED,
            FetchStatus.YOUTUBE_TRANSCRIPT,
            FetchStatus.YOUTUBE_META,
        }:
            continue
        if src.fetch_status is FetchStatus.DEAD:
            notes.append(f"Dead/parked URL not cited: {src.display_url}")
        elif src.fetch_note and src.fetch_status not in _REDDIT_UNUSABLE:
            label = src.title or src.display_url
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


def heuristic_judgment(
    entities: list[ExtractedEntity],
    sources: list[SourceRecord],
    requested: list[SourceType],
    *,
    iteration: int,
    max_iterations: int,
    question: str = "",
) -> Judgment:
    """Used when Gemini judge calls fail (503) so the run can still continue."""
    judgment = Judgment(
        enough=False,
        rationale="Judge model failed; coverage was computed from extracted entities.",
        entity_checks=_checks_from_entities(entities),
    )
    return enforce_coverage(
        judgment,
        entities,
        sources,
        requested,
        iteration=iteration,
        max_iterations=max_iterations,
        question=question,
    )


def _need_two_types(requested: list[SourceType], blocked_reddit: bool) -> bool:
    usable = [t for t in requested if not (t is SourceType.REDDIT and blocked_reddit)]
    return len(usable) >= 2


def _checks_from_entities(entities: list[ExtractedEntity]) -> list[EntityCheck]:
    return [
        EntityCheck(
            name=e.name,
            kind=e.kind,
            corroborated=e.corroborated,
            source_ids=list(e.source_ids),
            source_types=list(e.source_types),
        )
        for e in entities
        if e.kind != "area"
    ]


def _ensure_check(checks: list[EntityCheck], entity: ExtractedEntity, *, corroborated: bool) -> None:
    for check in checks:
        if check.name.lower() == entity.name.lower():
            check.corroborated = corroborated
            check.source_ids = list(entity.source_ids)
            check.source_types = list(entity.source_types)
            return
    checks.append(
        EntityCheck(
            name=entity.name,
            kind=entity.kind,
            corroborated=corroborated,
            source_ids=list(entity.source_ids),
            source_types=list(entity.source_types),
        )
    )


def _followups_for(
    uncorroborated: list[ExtractedEntity],
    missing: list[SourceType],
    blocked_reddit: bool,
    *,
    question: str = "",
) -> list[FollowupQuery]:
    out: list[FollowupQuery] = []
    topic = _topic_from_question(question)
    for entity in uncorroborated[:4]:
        out.append(
            FollowupQuery(
                query=f'"{entity.name}" review',
                source_type=SourceType.BLOG,
                gap=f"Second source type for {entity.name}",
            )
        )
        if not blocked_reddit:
            out.append(
                FollowupQuery(
                    query=f'"{entity.name}" site:reddit.com',
                    source_type=SourceType.REDDIT,
                    gap=f"Traveller mentions of {entity.name}",
                )
            )
    for source_type in missing:
        if source_type is SourceType.REDDIT and blocked_reddit:
            continue
        out.append(
            FollowupQuery(
                query=topic,
                source_type=source_type,
                gap=f"Missing {source_type.value} coverage",
            )
        )
    return out[:4]


def _topic_from_question(question: str) -> str:
    text = re.sub(r"\s+", " ", (question or "").strip())
    if not text:
        return "the topic"
    text = re.split(r"\bsearch\b", text, maxsplit=1, flags=re.I)[0]
    text = re.split(r"(?<!\d)[.?!](?!\d)", text, maxsplit=1)[0]
    text = text.strip(" ,;:-")
    return text[:90] or "the topic"


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
