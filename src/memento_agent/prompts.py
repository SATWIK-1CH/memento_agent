"""Prompts for plan / judge / write. Kept here so the loop stays easy to read."""

from __future__ import annotations

from memento_agent.types import Judgment, SourceRecord, SourceType

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
Enough means:
- The main parts of the question can be answered from readable evidence.
- Named hotels/places are corroborated across at least TWO source types when possible
  (e.g. a travel blog AND Reddit, or an official site AND a video title). Listicles
  often mislabel restaurants as hotels or use old names — treat a name that appears
  in only one listicle as uncorroborated.
- Dead, parked, or unfetched official URLs do not count as corroboration.
- Reddit search snippets ARE usable evidence when the thread itself was blocked;
  YouTube is title/description only, never a transcript.
- Requested source types were attempted; missing types are listed honestly.
Fill entity_checks for every hotel/area you would mention. If a name is uncorroborated,
add a follow-up query for its official/current name.
If not enough, propose 1 to 3 follow-up searches that close the biggest gaps.
Do not write the user-facing answer.
"""

WRITE_SYSTEM = """You are the writer for a personal research agent.
Write a structured research brief from the sources. Every claim must cite source ids
like S1, S2 — never invent a source id, never cite a source you were not given.
Rules:
- Prefer entities corroborated across 2+ source types. If only one listicle mentions a
  name, say so and do not present it as a confirmed hotel.
- Never cite a source with status=dead (parked or dead domain).
- YouTube: cite as "featured in video <title>" only. Do not claim room details, prices,
  or opinions from a video you did not watch. Do not use transcripts.
- Reddit: if status is blocked or snippet_only, cite the search snippet and say the
  thread could not be opened.
- If the question is about hotels / places, group items into tiers (budget, mid-range,
  splurge) when sources support it, and explain why.
- Be honest about unread pages, 406/timeouts, and name mismatches (renamed hotels).
- Do not pad. Thin evidence belongs in limitations and open_questions.
- Narrative fields may use [S1] inline citations.
"""


def plan_user_prompt(question: str, source_types: list[SourceType]) -> str:
    types = ", ".join(t.value for t in source_types)
    return (
        f"User question:\n{question}\n\n"
        f"Requested source types: {types}\n"
        "Produce the plan."
    )


def evidence_block(sources: list[SourceRecord], *, excerpt_limit: int = 4_000) -> str:
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
) -> str:
    types = ", ".join(t.value for t in source_types)
    coverage = _coverage_summary(sources, source_types)
    return (
        f"User question:\n{question}\n\n"
        f"Requested source types: {types}\n"
        f"Iteration {iteration} of {max_iterations}.\n"
        f"Coverage:\n{coverage}\n\n"
        f"Sources:\n{evidence_block(sources)}\n\n"
        "Judge whether this is enough."
    )


def write_user_prompt(
    question: str,
    source_types: list[SourceType],
    sources: list[SourceRecord],
    judgment: Judgment,
) -> str:
    types = ", ".join(t.value for t in source_types)
    return (
        f"User question:\n{question}\n\n"
        f"Requested source types: {types}\n"
        f"Critic rationale: {judgment.rationale}\n"
        f"Gaps the critic still sees: {judgment.gaps or 'none listed'}\n"
        f"Corroborated points: {judgment.corroborated_points or 'none listed'}\n\n"
        f"Sources you may cite:\n{evidence_block(sources)}\n\n"
        "Write the structured answer. Cite source ids in every item and narrative."
    )


def _coverage_summary(sources: list[SourceRecord], requested: list[SourceType]) -> str:
    lines: list[str] = []
    for source_type in requested:
        matching = [s for s in sources if s.source_type is source_type]
        readable = [
            s
            for s in matching
            if s.fetch_status.value in {"ok", "youtube_meta", "snippet_only"}
            and (s.excerpt or s.snippet)
        ]
        lines.append(
            f"- {source_type.value}: {len(matching)} hits, {len(readable)} readable"
        )
    return "\n".join(lines)
