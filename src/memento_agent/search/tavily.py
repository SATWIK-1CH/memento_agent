"""Tavily Search — default free-tier search backend."""

from __future__ import annotations

from typing import Any

from memento_agent.errors import ConfigError, QuotaExhaustedError, RateLimitError
from memento_agent.retry import call_with_retry
from memento_agent.sources import classify_url
from memento_agent.types import SearchHit, SourceType

# Researcher (free) plan: 1,000 credits/month; basic search = 1 credit; 100 RPM.
# https://docs.tavily.com/documentation/api-credits
# https://docs.tavily.com/documentation/rate-limits

_YOUTUBE_DOMAINS = ["youtube.com", "youtu.be"]
_REDDIT_DOMAINS = ["reddit.com"]
_SOCIAL_DOMAINS = ["youtube.com", "youtu.be", "reddit.com"]


class TavilySearchProvider:
    name = "tavily"

    def __init__(self, api_key: str, *, client: Any | None = None) -> None:
        self._api_key = api_key
        self._client = client

    def search(
        self,
        query: str,
        *,
        source_type: SourceType | None = None,
        max_results: int = 8,
    ) -> list[SearchHit]:
        if not self._api_key and self._client is None:
            raise ConfigError(
                "No Tavily API key. Set TAVILY_API_KEY (free Researcher plan, 1,000 "
                "credits/month, no credit card) at https://app.tavily.com"
            )
        kwargs: dict[str, Any] = {
            "query": _tavily_query(query, source_type),
            "search_depth": "basic",
            "max_results": max(1, min(max_results, 20)),
            "include_answer": False,
        }
        include, exclude = _domain_filters(source_type)
        if include:
            kwargs["include_domains"] = include
            kwargs["include_domains_mode"] = "restrict"
        if exclude:
            kwargs["exclude_domains"] = exclude
        if source_type is SourceType.REDDIT:
            # Same 1-credit basic search; raw_content survives Reddit HTTP 403s.
            # https://docs.tavily.com/documentation/api-reference/endpoint/search
            kwargs["include_raw_content"] = True

        payload = call_with_retry(lambda: self._call_search(kwargs))
        hits: list[SearchHit] = []
        for item in payload.get("results") or []:
            url = str(item.get("url") or "")
            if not url.startswith("http"):
                continue
            raw = str(item.get("raw_content") or "").strip()
            snippet = raw or str(item.get("content") or item.get("snippet") or "")
            if len(snippet) > 12_000:
                snippet = snippet[:12_000].rstrip() + "…"
            hits.append(
                SearchHit(
                    url=url,
                    title=str(item.get("title") or ""),
                    snippet=snippet,
                    source_type=classify_url(url, hinted=source_type),
                    query=query,
                )
            )
        return hits[:max_results]

    def extract(self, urls: list[str], *, query: str = "") -> dict[str, str]:
        """Tavily Extract (basic = 1 credit per 5 URLs) for pages the fetcher could not open."""
        urls = [u for u in urls if u.startswith("http")][:20]
        if not urls:
            return {}
        if not self._api_key and self._client is None:
            return {}

        def _call() -> dict[str, Any]:
            client = self._client_or_create()
            kwargs: dict[str, Any] = {"urls": urls, "extract_depth": "basic"}
            if query:
                kwargs["query"] = query
                kwargs["chunks_per_source"] = 3
            try:
                result = client.extract(**kwargs)
            except TypeError:
                kwargs.pop("chunks_per_source", None)
                kwargs.pop("query", None)
                result = client.extract(**kwargs)
            except Exception as exc:
                raise _map_tavily_error(exc) from exc
            return result if isinstance(result, dict) else {"results": list(result or [])}

        try:
            payload = call_with_retry(_call)
        except Exception:
            return {}
        out: dict[str, str] = {}
        for item in payload.get("results") or []:
            url = str(item.get("url") or "")
            content = str(item.get("raw_content") or item.get("content") or "").strip()
            if url and content:
                out[url] = content
        return out

    def _call_search(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        client = self._client_or_create()
        try:
            result = client.search(**kwargs)
        except TypeError:
            kwargs = dict(kwargs)
            kwargs.pop("include_domains_mode", None)
            result = client.search(**kwargs)
        except Exception as exc:
            raise _map_tavily_error(exc) from exc
        if not isinstance(result, dict):
            return {"results": list(result or [])}
        return result

    def _client_or_create(self) -> Any:
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self._api_key)
        return self._client


def _domain_filters(source_type: SourceType | None) -> tuple[list[str] | None, list[str] | None]:
    if source_type is SourceType.YOUTUBE:
        return list(_YOUTUBE_DOMAINS), None
    if source_type is SourceType.REDDIT:
        return list(_REDDIT_DOMAINS), None
    if source_type is SourceType.BLOG:
        return None, list(_SOCIAL_DOMAINS)
    return None, None


def _tavily_query(query: str, source_type: SourceType | None) -> str:
    q = query.strip()
    if source_type is SourceType.BLOG:
        lowered = q.lower()
        if not any(word in lowered for word in ("blog", "review", "article", "news")):
            q = f"{q} (blog OR review OR article)"
    return q


def _map_tavily_error(exc: BaseException) -> Exception:
    text = str(exc).lower()
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code == 401 or "invalid api key" in text or "unauthorized" in text:
        return ConfigError(
            "Tavily rejected TAVILY_API_KEY. Create a free key at https://app.tavily.com"
        )
    if code == 429 or "rate limit" in text or "excessive requests" in text:
        if "credit" in text or "quota" in text:
            return QuotaExhaustedError(
                "Tavily credits are exhausted. The free Researcher plan includes 1,000 "
                "credits/month (basic search = 1 credit). Wait for the monthly reset or "
                "check usage at https://app.tavily.com"
            )
        return RateLimitError(
            "Tavily rate-limited this request (free plan is 100 RPM). Wait a minute and retry."
        )
    if "credit" in text and ("exceed" in text or "insufficient" in text):
        return QuotaExhaustedError(
            "Tavily credits are exhausted. Free Researcher plan: 1,000 credits/month. "
            "https://docs.tavily.com/documentation/api-credits"
        )
    from memento_agent.errors import LLMError

    return LLMError(f"Tavily search failed: {exc}")
