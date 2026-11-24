"""Autonomous research loop. Independent of the CLI so a tray/voice UI can call it."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from memento_agent.config import Settings
from memento_agent.corroborate import (
    enforce_coverage,
    heuristic_judgment,
    merge_followups,
    reddit_unreachable,
)
from memento_agent.entities import (
    extract_entities,
    merge_missing_entities,
    scrub_citations,
    tidy_limitations,
    tidy_open_questions,
)
from memento_agent.errors import LLMError, RateLimitError
from memento_agent.events import Event, EventBus
from memento_agent.fetch import Fetcher
from memento_agent.llm import GeminiLLM
from memento_agent.preflight import preflight
from memento_agent.render import render_markdown
from memento_agent.search import build_search_provider
from memento_agent.sources import detect_source_types, is_reddit_thread
from memento_agent.store import Store
from memento_agent.types import (
    FetchStatus,
    PlannedQuery,
    ResearchAnswer,
    ResearchResult,
    SearchHit,
    SourceRecord,
    SourceType,
)
from memento_agent.youtube import YouTubeAnalyzer


class ResearchAgent:
    def __init__(
        self,
        settings: Settings,
        *,
        llm: Any,
        search: Any,
        fetcher: Fetcher,
        store: Store,
        youtube: Any | None = None,
        on_event: Callable[[Event], None] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.search = search
        self.fetcher = fetcher
        self.youtube = youtube
        self.store = store
        self.events = EventBus(on_event)
        self._sleeper = sleeper
        if hasattr(llm, "on_fallback"):
            llm.on_fallback = lambda msg: self.events.emit("fallback", msg)

    def run(self, question: str, *, source_types: list[SourceType] | None = None) -> ResearchResult:
        question = question.strip()
        if not question:
            raise ValueError("Question is empty.")
        types = source_types or detect_source_types(question)
        run_id = uuid.uuid4().hex[:12]
        self.store.start_run(
            run_id,
            question,
            types,
            model=self.llm.model if hasattr(self.llm, "model") else self.settings.model,
            search_model=self.settings.search_label,
        )
        self.events.emit("start", f"Run {run_id}", run_id=run_id, source_types=[t.value for t in types])

        iterations = 0
        searches_used = 0
        sources: list[SourceRecord] = []
        seen_urls: set[str] = set()
        all_queries: list[PlannedQuery] = []
        last_judgment = None

        try:
            self.events.emit("plan", "Breaking the question into sub-queries…")
            plan = self.llm.plan(question, types)
            pending = list(plan.sub_queries)
            all_queries.extend(pending)
            self.store.add_sub_queries(run_id, 0, pending)
            self.events.emit(
                "planned",
                f"{len(pending)} sub-queries",
                queries=[q.model_dump() for q in pending],
            )

            for iteration in range(1, self.settings.max_iterations + 1):
                if not pending:
                    break
                iterations = iteration
                batch = pending
                pending = []
                for query in batch:
                    if searches_used >= self.settings.max_searches:
                        self.events.emit("budget", "Search budget reached.")
                        break
                    self.events.emit(
                        "search",
                        f"[{query.source_type.value}] {query.query}",
                        iteration=iteration,
                    )
                    if searches_used and self.settings.search_delay_s:
                        self._sleeper(self.settings.search_delay_s)
                    hits: list[SearchHit] = self.search.search(
                        query.query,
                        source_type=query.source_type,
                        max_results=self.settings.max_pages_per_search + 2,
                    )
                    searches_used += 1
                    new_hits = [h for h in hits if _url_key(h.url) not in seen_urls]
                    for hit in new_hits[: self.settings.max_pages_per_search]:
                        seen_urls.add(_url_key(hit.url))
                        source_id = f"S{len(sources) + 1}"
                        record = self.fetcher.fetch(hit, source_id=source_id, iteration=iteration)
                        sources.append(record)
                        self.store.add_source(run_id, record)
                        self.events.emit(
                            "fetch",
                            f"{record.id} {record.fetch_status.value}: {record.title or record.url}",
                            source_id=record.id,
                            status=record.fetch_status.value,
                        )

                self._enrich_reddit(sources, question)
                if self.youtube is not None:
                    self.youtube.enrich(sources, question, events=self.events)
                for record in sources:
                    self.store.add_source(run_id, record)

                inventory = extract_entities(sources, question)
                self.events.emit("judge", f"Judging evidence after iteration {iteration}…")
                last_judgment = self._judge(
                    question,
                    sources,
                    types,
                    inventory,
                    iteration=iteration,
                )
                last_judgment = enforce_coverage(
                    last_judgment,
                    inventory,
                    sources,
                    types,
                    iteration=iteration,
                    max_iterations=self.settings.max_iterations,
                    question=question,
                )
                if searches_used >= self.settings.max_searches:
                    last_judgment.enough = True
                self.store.add_judgment(run_id, iteration, last_judgment)
                self.events.emit(
                    "judged",
                    last_judgment.rationale or ("enough" if last_judgment.enough else "need more"),
                    enough=last_judgment.enough,
                    gaps=last_judgment.gaps,
                )
                if last_judgment.enough or searches_used >= self.settings.max_searches:
                    break
                pending = merge_followups(
                    last_judgment,
                    sources,
                    entities=inventory,
                    blocked_reddit=reddit_unreachable(sources),
                )
                if pending:
                    all_queries.extend(pending)
                    self.store.add_sub_queries(run_id, iteration, pending)

            inventory = extract_entities(sources, question)
            citable = [s for s in sources if s.citable]
            if not citable:
                answer = ResearchAnswer(
                    summary="No citable sources were retrieved, so no factual answer is given.",
                    limitations=[
                        "Every candidate URL failed, was parked/dead, or returned no usable snippet."
                    ],
                    open_questions=[question],
                )
            else:
                self.events.emit("write", "Writing a cited answer…")
                if last_judgment is None:
                    last_judgment = self._judge(
                        question,
                        sources,
                        types,
                        inventory,
                        iteration=max(iterations, 1),
                    )
                    last_judgment = enforce_coverage(
                        last_judgment,
                        inventory,
                        sources,
                        types,
                        iteration=max(iterations, 1),
                        max_iterations=self.settings.max_iterations,
                        question=question,
                    )
                    self.store.add_judgment(run_id, max(iterations, 1), last_judgment)
                answer = self._write(question, sources, types, last_judgment, inventory)

            markdown = render_markdown(question, answer, sources, judgment=last_judgment)
            self.store.complete_run(
                run_id,
                answer=answer,
                markdown=markdown,
                iterations=iterations,
                searches_used=searches_used,
            )
            self.events.emit("done", "Research complete.", run_id=run_id)
            return ResearchResult(
                run_id=run_id,
                question=question,
                source_types=types,
                answer=answer,
                markdown=markdown,
                sources=sources,
                sub_queries=all_queries,
                iterations=iterations,
                searches_used=searches_used,
            )
        except Exception as exc:
            self.store.fail_run(run_id, str(exc), iterations=iterations, searches_used=searches_used)
            raise

    def _judge(self, question, sources, types, inventory, *, iteration: int):
        try:
            return self.llm.judge(
                question,
                sources,
                types,
                iteration=iteration,
                max_iterations=self.settings.max_iterations,
                entities=inventory,
            )
        except TypeError:
            return self.llm.judge(
                question,
                sources,
                types,
                iteration=iteration,
                max_iterations=self.settings.max_iterations,
            )
        except (RateLimitError, LLMError) as exc:
            self.events.emit(
                "fallback",
                f"Judge failed ({type(exc).__name__}: {exc}); using coverage heuristic.",
            )
            return heuristic_judgment(
                inventory,
                sources,
                types,
                iteration=iteration,
                max_iterations=self.settings.max_iterations,
                question=question,
            )

    def _write(self, question, sources, types, judgment, inventory):
        try:
            answer = self.llm.write(question, sources, types, judgment, entities=inventory)
        except TypeError:
            answer = self.llm.write(question, sources, types, judgment)
        answer = merge_missing_entities(answer, inventory, sources)
        answer = scrub_citations(answer, sources)
        answer.limitations = tidy_limitations(answer.limitations)
        answer.open_questions = tidy_open_questions(answer.open_questions)
        return answer

    def _enrich_reddit(self, sources: list[SourceRecord], question: str) -> None:
        if not hasattr(self.search, "extract"):
            self.events.emit("fetch", "Tavily extract skipped (search provider has no extract).")
            return
        blocked = [
            s
            for s in sources
            if s.source_type is SourceType.REDDIT
            and s.fetch_status in {FetchStatus.BLOCKED, FetchStatus.SNIPPET_ONLY, FetchStatus.FAILED}
            and is_reddit_thread(s.display_url or s.url)
        ]
        if not blocked:
            self.events.emit("fetch", "Tavily extract: no blocked Reddit threads to retry.")
            return
        urls = [s.display_url or s.url for s in blocked]
        self.events.emit("fetch", f"Tavily extract for {len(urls)} blocked Reddit thread(s)…")
        try:
            mapping = self.search.extract(urls, query=question) or {}
        except Exception as exc:
            self.events.emit("fetch", f"Tavily extract failed ({type(exc).__name__}: {exc}).")
            return
        if not mapping:
            self.events.emit("fetch", "Tavily extract returned no Reddit content.")
            return
        for source in blocked:
            text = mapping.get(source.display_url) or mapping.get(source.url) or ""
            if len(text) <= len(source.evidence_text):
                self.events.emit(
                    "fetch",
                    f"{source.id} Tavily extract empty or not longer than snippet.",
                )
                continue
            source.excerpt = text[: self.settings.excerpt_chars]
            source.fetch_status = FetchStatus.OK
            source.fetch_note = "Opened via Tavily extract after Reddit HTTP block."
            self.events.emit("fetch", f"{source.id} extracted via Tavily: {source.title or source.url}")


def research(
    question: str,
    *,
    settings: Settings | None = None,
    store: Store | None = None,
    on_event: Callable[[Event], None] | None = None,
    llm: Any | None = None,
    search: Any | None = None,
    fetcher: Fetcher | None = None,
    youtube: Any | None = None,
) -> ResearchResult:
    """Public entry point used by the CLI and (later) a Windows tray/voice front end."""
    settings = settings or Settings.from_env()
    store = store or Store(settings.db_path)
    events = EventBus(on_event)
    llm = llm or GeminiLLM(settings)
    if search is None:
        preflight(settings, llm, events=events)
        search = build_search_provider(settings, llm)
    owns_fetcher = fetcher is None
    fetcher = fetcher or Fetcher(
        timeout_s=settings.request_timeout_s,
        delay_s=settings.fetch_delay_s,
        excerpt_chars=settings.excerpt_chars,
    )
    if youtube is None and hasattr(llm, "watch_video"):
        youtube = YouTubeAnalyzer(
            llm,
            max_videos=settings.max_videos,
            excerpt_chars=settings.excerpt_chars,
            watch_gap_s=settings.watch_gap_s,
        )
    agent = ResearchAgent(
        settings,
        llm=llm,
        search=search,
        fetcher=fetcher,
        store=store,
        youtube=youtube,
        on_event=on_event,
    )
    try:
        return agent.run(question)
    finally:
        if owns_fetcher:
            fetcher.close()


def _url_key(url: str) -> str:
    return url.split("#", 1)[0].rstrip("/").lower()
