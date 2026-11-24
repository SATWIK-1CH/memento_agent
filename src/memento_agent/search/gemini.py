"""Gemini Grounding with Google Search, exposed as a SearchProvider."""

from __future__ import annotations

import re
from typing import Any

from memento_agent.retry import call_with_retry
from memento_agent.sources import classify_url, steer_query
from memento_agent.types import SearchHit, SourceType

_URL_RE = re.compile(r"https?://[^\s)>\]]+", re.I)


class GeminiSearchProvider:
    """Uses Gemini's google_search tool and returns grounding chunks as hits.

    The model is *not* asked to write the user's final answer here. We only want
    URLs, titles, and snippets so the agent loop can fetch and judge itself.
    """

    name = "gemini-google-search"

    def __init__(self, llm: Any) -> None:
        # ``llm`` is a GeminiLLM (or a test double with generate_search).
        self._llm = llm

    def search(
        self,
        query: str,
        *,
        source_type: SourceType | None = None,
        max_results: int = 8,
    ) -> list[SearchHit]:
        steered = steer_query(query, source_type) if source_type else query
        prompt = (
            "Use Google Search to find the most relevant sources for this query. "
            "Return a compact bullet list of results with title and URL only. "
            "Do not write a full research answer.\n\n"
            f"Source type wanted: {source_type.value if source_type else 'any'}\n"
            f"Query: {steered}"
        )
        response = call_with_retry(lambda: self._llm.generate_search(prompt), grounded=True)
        hits = hits_from_response(response, query=query, source_type=source_type)
        return hits[:max_results]


def hits_from_response(
    response: Any,
    *,
    query: str,
    source_type: SourceType | None,
) -> list[SearchHit]:
    """Extract unique SearchHits from a generate_content response."""
    hits: list[SearchHit] = []
    seen: set[str] = set()

    def add(url: str, title: str, snippet: str) -> None:
        url = url.rstrip(".,);")
        key = url.split("#", 1)[0]
        if not url.startswith("http") or key in seen:
            return
        if "vertexaisearch.cloud.google.com" in url and not title:
            title = title or "Google Search result"
        seen.add(key)
        classified = classify_url(url, hinted=source_type)
        hits.append(
            SearchHit(
                url=url,
                title=title.strip(),
                snippet=snippet.strip(),
                source_type=classified,
                query=query,
            )
        )

    candidate = _first_candidate(response)
    metadata = getattr(candidate, "grounding_metadata", None) if candidate else None
    if metadata is None and isinstance(candidate, dict):
        metadata = candidate.get("grounding_metadata") or candidate.get("groundingMetadata")

    chunks = _attr(metadata, "grounding_chunks") or _attr(metadata, "groundingChunks") or []
    supports = _attr(metadata, "grounding_supports") or _attr(metadata, "groundingSupports") or []
    snippets_by_index = _snippets_from_supports(supports)

    for index, chunk in enumerate(chunks):
        web = _attr(chunk, "web")
        uri = _attr(web, "uri") or _attr(web, "url") or ""
        title = _attr(web, "title") or ""
        snippet = snippets_by_index.get(index, "")
        if uri:
            add(str(uri), str(title), str(snippet))

    text = getattr(response, "text", None) or ""
    if not hits:
        for url in _URL_RE.findall(text):
            add(url, "", "")

    if hits and text:
        # Attach a short snippet from the model's list when chunks had no support text.
        for hit in hits:
            if not hit.snippet:
                hit.snippet = _snippet_near_url(text, hit.url)

    return hits


def _first_candidate(response: Any) -> Any:
    candidates = getattr(response, "candidates", None)
    if not candidates and isinstance(response, dict):
        candidates = response.get("candidates")
    if not candidates:
        return None
    return candidates[0]


def _attr(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _snippets_from_supports(supports: Any) -> dict[int, str]:
    out: dict[int, str] = {}
    if not supports:
        return out
    for support in supports:
        segment = _attr(support, "segment") or {}
        text = _attr(segment, "text") or ""
        indices = (
            _attr(support, "grounding_chunk_indices")
            or _attr(support, "groundingChunkIndices")
            or []
        )
        for index in indices:
            try:
                out[int(index)] = str(text)
            except (TypeError, ValueError):
                continue
    return out


def _snippet_near_url(text: str, url: str) -> str:
    idx = text.find(url)
    if idx < 0:
        return ""
    start = max(0, idx - 80)
    end = min(len(text), idx + len(url) + 80)
    return re.sub(r"\s+", " ", text[start:end]).strip()[:240]
