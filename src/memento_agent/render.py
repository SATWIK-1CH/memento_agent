"""Turn a structured answer + sources into cited markdown."""

from __future__ import annotations

from memento_agent.corroborate import limitations_from_sources, summarize_entity_checks
from memento_agent.types import (
    FetchStatus,
    Judgment,
    ResearchAnswer,
    SourceRecord,
    SourceType,
)


def render_markdown(
    question: str,
    answer: ResearchAnswer,
    sources: list[SourceRecord],
    *,
    judgment: Judgment | None = None,
) -> str:
    source_by_id = {s.id: s for s in sources}
    lines: list[str] = [f"# {question.strip()}", "", answer.summary.strip(), ""]

    for section in answer.sections:
        lines.append(f"## {section.title}")
        lines.append("")
        if section.narrative.strip():
            lines.append(_with_inline_cites(section.narrative.strip(), source_by_id))
            lines.append("")
        for item in section.items:
            cites = _cite_list(item.source_ids, source_by_id)
            title = item.name
            if item.tier:
                title = f"{item.name} ({item.tier})"
            lines.append(f"- **{title}** — {_with_inline_cites(item.why, source_by_id)} {cites}".rstrip())
        if section.items:
            lines.append("")

    limitations = list(answer.limitations)
    for note in limitations_from_sources(sources):
        if note not in limitations:
            limitations.append(note)
    if limitations:
        lines.append("## Limitations")
        lines.append("")
        for note in limitations:
            lines.append(f"- {note}")
        lines.append("")

    if answer.open_questions:
        lines.append("## Open questions")
        lines.append("")
        for q in answer.open_questions:
            lines.append(f"- {q}")
        lines.append("")

    if judgment and judgment.entity_checks:
        lines.append("## Entity checks")
        lines.append("")
        for line in summarize_entity_checks(judgment.entity_checks):
            lines.append(f"- {line}")
        lines.append("")

    lines.append("## Sources")
    lines.append("")
    for source_type in SourceType:
        group = [s for s in sources if s.source_type is source_type]
        if not group:
            continue
        lines.append(f"### {source_type.label}")
        lines.append("")
        for src in group:
            lines.append(_source_line(src))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _source_line(src: SourceRecord) -> str:
    title = src.title or src.display_url
    extra = ""
    if src.fetch_status is FetchStatus.YOUTUBE_META:
        extra = " — featured in this video (title/description only; not watched)"
    elif src.fetch_status is FetchStatus.DEAD:
        extra = " — dead/parked; not used as a live citation"
    elif src.fetch_status is FetchStatus.BLOCKED:
        extra = " — page blocked; search snippet only"
    elif src.fetch_status is FetchStatus.SNIPPET_ONLY:
        extra = " — search snippet only"
    elif src.fetch_status is FetchStatus.FAILED:
        extra = " — fetch failed; search snippet only"
    return f"- **[{src.id}]** [{title}]({src.display_url}){extra}"


def _cite_list(source_ids: list[str], source_by_id: dict[str, SourceRecord]) -> str:
    parts: list[str] = []
    for sid in source_ids:
        src = source_by_id.get(sid)
        if src is None or not src.citable:
            continue
        parts.append(f"[{sid}]({src.display_url})")
    return " ".join(parts)


def _with_inline_cites(text: str, source_by_id: dict[str, SourceRecord]) -> str:
    """Leave [S1] markers in the narrative; they match the Sources list."""
    return text
