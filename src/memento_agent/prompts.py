"""Prompts for plan / judge / write. Kept here so the loop stays easy to read."""

from __future__ import annotations

from memento_agent.entities import inventory_block
from memento_agent.types import ExtractedEntity, Judgment, SourceRecord, SourceType

PLAN_SYSTEM = """You are the planner for a personal research agent.
Break the user's question into a small set of web search sub-queries.
Rules:
- Honour requested source types (youtube, reddit, blog, web). Cover each requested type.
- 3 to 6 sub-queries. Each query should be something you would type into Google.
- Prefer specific, comparable queries (hotel names + area, "best beaches South Goa", etc.).
- For Reddit, prefer topic-specific subreddits when you know them (e.g. r/goatravel for Goa
  trips) over giant subs like r/travel or r/india.
- For hotels/places, include at least one query aimed at official sites or current names
  (renames and dead domains are common).
- Do not invent facts. You are only planning searches.
- Source type "blog" means travel blogs, magazines, and news sites.
"""

JUDGE_SYSTEM = """You are the critic for a personal research agent.
Decide whether the collected sources are enough to write a cited answer.
Enough means ALL of:
- The main parts of the question can be answered from readable evidence.
- Every stay in the entity inventory that looks like a hotel is corroborated across at
  least TWO source types when two types were requested (blog+reddit, blog+youtube, etc.).
  A name that appears in only one listicle or only YouTube is NOT enough.
- Follow-up queries exist for those single-type names (per-hotel review, per-area query).
- Dead, parked, or unfetched official URLs do not count as corroboration.
- Reddit search snippets ARE usable evidence when the thread itself was blocked. If every
  Reddit fetch was blocked, do not keep listing reddit as missing — note it once.
- YouTube youtube_watched / youtube_transcript / youtube_meta as labelled.
enough=false unless this is the last iteration or the search budget is gone.
Fill entity_checks for EVERY name in the inventory. Propose 1 to 3 targeted follow-ups.
Do not write the user-facing answer.
"""

VIDEO_WATCH_PROMPT = """Watch this public YouTube video (title: {title}).
Extract ONLY what is actually shown on screen or said out loud. Do not use the title,
description, comments, or general knowledge to fill gaps.
Return JSON with:
- summary: 2-4 sentences of what the video actually covers
- stays_shown: hotel / stay / place names that appear on screen or are named by the speaker
- prices_stated: prices only if spoken or shown (include currency and what it is for)
- pros: positive points the speaker or on-screen text actually makes
- cons: negative points actually made
- timestamps: notable moments as mm:ss — label (stay name, price, beach walk, etc.)
- caveats: anything unclear, sponsored, old, or not shown
If the video is unrelated or you cannot access it, put that in caveats and leave lists empty.
"""

WRITE_SYSTEM = """You are the writer for a personal research agent.
Write a structured research brief from the sources. Every claim must cite source ids
like S1, S2 — never invent a source id, never cite a source you were not given.
Rules:
- Prefer entities corroborated across 2+ source types. If only one listicle mentions a
  name, say so and do not present it as a confirmed hotel.
- Never cite a source with status=dead (parked or dead domain).
- YouTube: honour fetch_status.
  youtube_watched — you may cite stays shown, prices spoken/on-screen, and timestamps.
  youtube_transcript — you may cite captions, and must say they are captions not a watch.
  youtube_meta — cite as "featured in video <title>" only. Do not invent room details.
- Reddit: if status is blocked or snippet_only, cite the search snippet and say the
  thread could not be opened.
- If the question is about hotels / places, group items into tiers (budget, mid-range,
  splurge) when sources support it, and explain why.
- Include EVERY stay in the entity inventory. If evidence is thin, say so, but do not omit
  the name. Include stated prices with the source id and a date caveat.
- Cite a source for a claim only if that source's excerpt/title actually names the entity.
  Do not attach another hotel's description to this one.
- Be honest about unread pages, 406/timeouts, and name mismatches (renamed hotels).
- Do not pad. Thin evidence belongs in limitations. Open questions should be specific
  (a missing price, a rename) — not "needs more research".
- Narrative fields may use [S1] inline citations.
"""


def plan_user_prompt(question: str, source_types: list[SourceType]) -> str:
    types = ", ".join(t.value for t in source_types)
    return (
        f"User question:\n{question}\n\n"
        f"Requested source types: {types}\n"
        "Produce the plan."
    )


def evidence_block(sources: list[SourceRecord], *, excerpt_limit: int = 1_200) -> str:
    chunks: list[str] = []
    for src in sources:
        excerpt = src.excerpt or src.snippet
        if len(excerpt) > excerpt_limit:
            excerpt = excerpt[:excerpt_limit].rstrip() + "…"
        chunks.append(
            "\n".join(
                [
                    f"[{src.id}] type={src.source_type.value} status={src.fetch_status.value}",
                    f"title: {src.title or '(no title)'}",
                    f"url: {src.display_url}",
                    f"query: {src.query}",
                    f"fetch_note: {src.fetch_note or 'ok'}",
                    "excerpt:",
                    excerpt or "(empty)",
                ]
            )
        )
    return "\n\n----\n\n".join(chunks) if chunks else "(no sources yet)"


def judge_user_prompt(
    question: str,
    source_types: list[SourceType],
    sources: list[SourceRecord],
    iteration: int,
    max_iterations: int,
    *,
    entities: list[ExtractedEntity] | None = None,
    excerpt_limit: int = 1_200,
) -> str:
    types = ", ".join(t.value for t in source_types)
    coverage = _coverage_summary(sources, source_types)
    last = "LAST ITERATION — you may set enough=true with remaining gaps listed." if iteration >= max_iterations else (
        "Not the last iteration: enough=false if any stay is still single-source-type "
        "or a requested type (other than fully-blocked Reddit) is missing."
    )
    compact = _compact_sources(sources)
    return (
        f"User question:\n{question}\n\n"
        f"Requested source types: {types}\n"
        f"Iteration {iteration} of {max_iterations}. {last}\n"
        f"Coverage:\n{coverage}\n\n"
        f"{inventory_block(entities or [])}\n\n"
        f"Sources:\n{evidence_block(compact, excerpt_limit=excerpt_limit)}\n\n"
        "Judge whether this is enough. Fill entity_checks from the inventory."
    )


def write_user_prompt(
    question: str,
    source_types: list[SourceType],
    sources: list[SourceRecord],
    judgment: Judgment,
    *,
    entities: list[ExtractedEntity] | None = None,
) -> str:
    types = ", ".join(t.value for t in source_types)
    return (
        f"User question:\n{question}\n\n"
        f"Requested source types: {types}\n"
        f"Critic rationale: {judgment.rationale}\n"
        f"Gaps the critic still sees: {judgment.gaps or 'none listed'}\n"
        f"Corroborated points: {judgment.corroborated_points or 'none listed'}\n\n"
        f"{inventory_block(entities or [])}\n\n"
        f"Sources you may cite:\n{evidence_block(sources, excerpt_limit=2_400)}\n\n"
        "Write the structured answer. Include every inventory stay. Cite source ids "
        "only when that source names the entity. Include stated prices with a date caveat. "
        "For YouTube, say whether the video was watched, captions-only, or title-only."
    )


def _coverage_summary(sources: list[SourceRecord], requested: list[SourceType]) -> str:
    lines: list[str] = []
    for source_type in requested:
        matching = [s for s in sources if s.source_type is source_type]
        readable = [
            s
            for s in matching
            if s.fetch_status.value
            in {
                "ok",
                "youtube_watched",
                "youtube_transcript",
                "youtube_meta",
                "snippet_only",
            }
            and (s.excerpt or s.snippet or s.title)
        ]
        lines.append(
            f"- {source_type.value}: {len(matching)} hits, {len(readable)} readable"
        )
    return "\n".join(lines)


def _compact_sources(sources: list[SourceRecord], *, limit: int = 12) -> list[SourceRecord]:
    """Keep a type-diverse slice so flash-lite judge prompts stay small."""
    if len(sources) <= limit:
        return sources
    picked: list[SourceRecord] = []
    seen: set[str] = set()
    for source_type in SourceType:
        for src in sources:
            if src.source_type is not source_type or src.id in seen:
                continue
            if not src.citable:
                continue
            picked.append(src)
            seen.add(src.id)
            if len(picked) >= limit:
                return picked
    for src in sources:
        if src.id in seen:
            continue
        picked.append(src)
        if len(picked) >= limit:
            break
    return picked
