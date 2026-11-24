"""Fetch and extract page text, with honest fallbacks for YouTube and Reddit.

Practical notes from real travel-research fetches:
- Prefer GET with a browser User-Agent. HEAD often 403s; some hotel sites 406
  unless Accept looks like a browser.
- Reddit threads and .json/old.reddit frequently time out or hit a login wall.
  Search snippets stay useful when the thread cannot be opened.
- Official hotel domains are sometimes parked or dead — validate before citing.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import trafilatura

from memento_agent.sources import classify_url
from memento_agent.types import FetchStatus, SearchHit, SourceRecord, SourceType

# Browser-like GET. A custom bot UA plus HEAD requests is what 403/406s many
# hotel and Reddit pages in practice.
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "no-cache",
}

_PARKED_TITLE = re.compile(
    r"parked|domain for sale|buy this domain|this domain is for sale|"
    r"domain has expired|website is for sale|coming soon|"
    r"account suspended|default web page",
    re.I,
)
_PARKED_BODY = re.compile(
    r"this domain(?: name)? is (?:parked|for sale)|buy this domain|"
    r"sedo domain parking|hugedomains|godaddy parking|"
    r"the domain is registered|this webpage is parked|"
    r"future home of|domain may be for sale",
    re.I,
)


class Fetcher:
    def __init__(
        self,
        *,
        timeout_s: float = 20.0,
        delay_s: float = 0.2,
        excerpt_chars: int = 8_000,
        client: httpx.Client | None = None,
        sleeper: Any = time.sleep,
    ) -> None:
        self._owns_client = client is None
        self._timeout_s = timeout_s
        self._client = client or httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(timeout_s, connect=min(8.0, timeout_s)),
            headers=DEFAULT_HEADERS,
        )
        self._delay_s = delay_s
        self._excerpt_chars = excerpt_chars
        self._sleeper = sleeper
        self._last_fetch_at = 0.0

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def fetch(self, hit: SearchHit, *, source_id: str, iteration: int) -> SourceRecord:
        self._throttle()
        source_type = hit.source_type or classify_url(hit.url)
        if source_type is SourceType.YOUTUBE:
            return self._fetch_youtube(hit, source_id=source_id, iteration=iteration)
        if source_type is SourceType.REDDIT:
            return self._fetch_reddit(hit, source_id=source_id, iteration=iteration)
        return self._fetch_html(hit, source_id=source_id, iteration=iteration, source_type=source_type)

    def _throttle(self) -> None:
        if self._delay_s <= 0:
            return
        now = time.monotonic()
        wait = self._delay_s - (now - self._last_fetch_at)
        if wait > 0:
            self._sleeper(wait)
        self._last_fetch_at = time.monotonic()

    def _record(
        self,
        hit: SearchHit,
        *,
        source_id: str,
        iteration: int,
        source_type: SourceType,
        status: FetchStatus,
        title: str = "",
        excerpt: str = "",
        note: str = "",
        final_url: str = "",
    ) -> SourceRecord:
        excerpt = excerpt.strip()
        if len(excerpt) > self._excerpt_chars:
            excerpt = excerpt[: self._excerpt_chars].rstrip() + "…"
        return SourceRecord(
            id=source_id,
            url=hit.url,
            final_url=final_url or hit.url,
            title=title or hit.title,
            source_type=source_type,
            snippet=hit.snippet,
            fetch_status=status,
            fetch_note=note,
            excerpt=excerpt or hit.snippet,
            query=hit.query,
            iteration=iteration,
        )

    def _get(self, url: str, *, timeout: float | None = None, headers: dict[str, str] | None = None) -> httpx.Response:
        """GET only (never HEAD). Retry once on 406 with a simpler Accept."""
        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = httpx.Timeout(timeout, connect=min(6.0, timeout))
        if headers:
            kwargs["headers"] = headers
        response = self._client.get(url, **kwargs)
        if response.status_code == 406:
            retry_headers = {**(headers or {}), "Accept": "*/*"}
            response = self._client.get(url, headers=retry_headers, **{k: v for k, v in kwargs.items() if k != "headers"})
        return response

    def _fetch_html(
        self,
        hit: SearchHit,
        *,
        source_id: str,
        iteration: int,
        source_type: SourceType,
    ) -> SourceRecord:
        try:
            response = self._get(hit.url)
        except httpx.TimeoutException:
            return self._record(
                hit,
                source_id=source_id,
                iteration=iteration,
                source_type=source_type,
                status=FetchStatus.FAILED,
                excerpt=hit.snippet,
                note="Timed out fetching the page; used the search snippet only.",
            )
        except httpx.HTTPError as exc:
            return self._record(
                hit,
                source_id=source_id,
                iteration=iteration,
                source_type=source_type,
                status=FetchStatus.FAILED,
                excerpt=hit.snippet,
                note=f"Could not fetch page: {exc}",
            )
        return self._record_from_response(hit, response, source_id=source_id, iteration=iteration, source_type=source_type)

    def _record_from_response(
        self,
        hit: SearchHit,
        response: httpx.Response,
        *,
        source_id: str,
        iteration: int,
        source_type: SourceType,
    ) -> SourceRecord:
        final_url = str(response.url)
        classified = classify_url(final_url, hinted=source_type)
        if response.status_code in {401, 403, 429, 451}:
            return self._record(
                hit,
                source_id=source_id,
                iteration=iteration,
                source_type=classified,
                status=FetchStatus.BLOCKED,
                title=hit.title,
                excerpt=hit.snippet,
                note=f"Site returned HTTP {response.status_code}; used the search snippet only.",
                final_url=final_url,
            )
        if response.status_code >= 400:
            return self._record(
                hit,
                source_id=source_id,
                iteration=iteration,
                source_type=classified,
                status=FetchStatus.FAILED,
                excerpt=hit.snippet,
                note=f"HTTP {response.status_code} fetching page; used the search snippet only.",
                final_url=final_url,
            )
        html = response.text or ""
        title = _html_title(html) or hit.title
        if looks_parked(html, title=title, final_url=final_url):
            return self._record(
                hit,
                source_id=source_id,
                iteration=iteration,
                source_type=classified,
                status=FetchStatus.DEAD,
                title=title,
                excerpt="",
                note=(
                    "URL looks parked or dead (for-sale / expired domain page). "
                    "Not used as a live citation."
                ),
                final_url=final_url,
            )
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
            url=final_url,
        )
        if extracted and extracted.strip():
            return self._record(
                hit,
                source_id=source_id,
                iteration=iteration,
                source_type=classified,
                status=FetchStatus.OK,
                title=title,
                excerpt=extracted,
                final_url=final_url,
            )
        return self._record(
            hit,
            source_id=source_id,
            iteration=iteration,
            source_type=classified,
            status=FetchStatus.SNIPPET_ONLY,
            title=title,
            excerpt=hit.snippet,
            note="Could not extract article text; used the search snippet only.",
            final_url=final_url,
        )

    def _fetch_youtube(self, hit: SearchHit, *, source_id: str, iteration: int) -> SourceRecord:
        watch_url = _canonical_youtube_url(hit.url) or hit.url
        title = hit.title
        description = ""
        extra_note = ""
        try:
            oembed = self._client.get(
                "https://www.youtube.com/oembed",
                params={"url": watch_url, "format": "json"},
                timeout=httpx.Timeout(min(12.0, self._timeout_s), connect=6.0),
            )
            if oembed.status_code == 200:
                payload = oembed.json()
                title = payload.get("title") or title
                author = payload.get("author_name") or ""
                if author:
                    description = f"Channel: {author}"
            elif oembed.status_code in {401, 403, 429}:
                extra_note = f"YouTube oEmbed returned HTTP {oembed.status_code}."
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            extra_note = f"YouTube oEmbed failed: {exc}"

        try:
            page = self._get(watch_url, timeout=min(15.0, self._timeout_s))
            if page.status_code == 200:
                html = page.text or ""
                title = title or _meta_content(html, "og:title") or _html_title(html)
                og = _meta_content(html, "og:description") or _meta_content(html, "description")
                if og:
                    description = f"{description}\n{og}".strip() if description else og
            elif page.status_code in {401, 403, 429}:
                extra_note = extra_note or f"YouTube watch page returned HTTP {page.status_code}."
        except httpx.HTTPError as exc:
            extra_note = extra_note or f"YouTube watch page failed: {exc}"

        excerpt_parts = [p for p in (description, hit.snippet) if p]
        excerpt = "\n\n".join(excerpt_parts)
        note = (
            "YouTube video was not watched and no transcript was used; "
            "citation is title/description only (featured in this video)."
        )
        if extra_note:
            note = f"{note} {extra_note}"
        return self._record(
            hit,
            source_id=source_id,
            iteration=iteration,
            source_type=SourceType.YOUTUBE,
            status=FetchStatus.YOUTUBE_META,
            title=title,
            excerpt=excerpt,
            note=note,
            final_url=watch_url,
        )

    def _fetch_reddit(self, hit: SearchHit, *, source_id: str, iteration: int) -> SourceRecord:
        """Try a browser GET of the thread; fall back to the search snippet.

        old.reddit.com and .json often time out or hit a login wall. Snippets of
        comments (e.g. r/goatravel) are still useful and should not be discarded.
        """
        reddit_timeout = min(12.0, self._timeout_s)
        html_urls = _reddit_html_urls(hit.url)
        last_error = ""
        for url in html_urls:
            try:
                response = self._get(url, timeout=reddit_timeout)
            except httpx.TimeoutException:
                last_error = f"Timed out fetching {url}"
                continue
            except httpx.HTTPError as exc:
                last_error = str(exc)
                continue
            if response.status_code in {401, 403, 429, 451}:
                last_error = f"HTTP {response.status_code} from {url}"
                continue
            if response.status_code >= 400:
                last_error = f"HTTP {response.status_code} from {url}"
                continue
            html = response.text or ""
            if _looks_like_reddit_login(html):
                last_error = f"Login wall at {url}"
                continue
            extracted = trafilatura.extract(
                html,
                include_comments=True,
                include_tables=False,
                favor_recall=True,
                url=str(response.url),
            )
            title = _html_title(html) or hit.title
            if extracted and len(extracted.strip()) > 80:
                return self._record(
                    hit,
                    source_id=source_id,
                    iteration=iteration,
                    source_type=SourceType.REDDIT,
                    status=FetchStatus.OK,
                    title=title,
                    excerpt=extracted,
                    final_url=str(response.url),
                )

        json_url = _reddit_json_url(hit.url)
        try:
            response = self._get(
                json_url,
                timeout=reddit_timeout,
                headers={**DEFAULT_HEADERS, "Accept": "application/json,text/plain,*/*;q=0.8"},
            )
            if response.status_code == 200:
                payload = response.json()
                title, excerpt = _reddit_excerpt(payload, fallback=hit.snippet)
                if excerpt and excerpt != hit.snippet:
                    return self._record(
                        hit,
                        source_id=source_id,
                        iteration=iteration,
                        source_type=SourceType.REDDIT,
                        status=FetchStatus.OK,
                        title=title or hit.title,
                        excerpt=excerpt,
                        final_url=str(response.url),
                    )
            else:
                last_error = last_error or f"Reddit JSON HTTP {response.status_code}"
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            last_error = last_error or f"Reddit JSON failed: {exc}"

        note = (
            "Could not open the Reddit thread (timeout, login wall, or block). "
            "Used the search snippet only; comment context may be incomplete."
        )
        if last_error:
            note = f"{note} ({last_error})"
        status = FetchStatus.BLOCKED if "HTTP 403" in last_error or "Login wall" in last_error else FetchStatus.SNIPPET_ONLY
        return self._record(
            hit,
            source_id=source_id,
            iteration=iteration,
            source_type=SourceType.REDDIT,
            status=status,
            excerpt=hit.snippet,
            note=note,
        )


def looks_parked(html: str, *, title: str = "", final_url: str = "") -> bool:
    """True when the page is a parked/expired domain rather than a live site."""
    if title and _PARKED_TITLE.search(title):
        return True
    # Parking pages are usually tiny and full of sale copy.
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    if _PARKED_BODY.search(text[:4_000]):
        return True
    if final_url:
        host = urlparse(final_url).netloc.lower()
        if any(p in host for p in ("hugedomains.com", "sedoparking.com", "parkingcrew.net")):
            return True
    return False


def _html_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    if not match:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", match.group(1))).strip()


def _meta_content(html: str, name: str) -> str:
    patterns = [
        rf'<meta[^>]+(?:property|name)=["\']{re.escape(name)}["\'][^>]+content=["\'](.*?)["\']',
        rf'<meta[^>]+content=["\'](.*?)["\'][^>]+(?:property|name)=["\']{re.escape(name)}["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.I | re.S)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def _canonical_youtube_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        video_id = parsed.path.strip("/")
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id.split('/')[0]}"
    query = parse_qs(parsed.query)
    video_id = (query.get("v") or [None])[0]
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return url


def _reddit_html_urls(url: str) -> list[str]:
    parsed = urlparse(url)
    path = parsed.path or "/"
    if path.endswith(".json"):
        path = path[: -len(".json")]
    path = path.rstrip("/") or "/"
    # Public www GET first (browser UA); old.reddit as a second try.
    return [
        f"https://www.reddit.com{path}",
        f"https://old.reddit.com{path}",
    ]


def _reddit_json_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path.endswith(".json"):
        json_path = path
    else:
        json_path = f"{path}.json"
    return f"https://old.reddit.com{json_path}?raw_json=1"


def _looks_like_reddit_login(html: str) -> bool:
    lower = html.lower()
    if "log in" in lower and "reddit" in lower and len(html) < 8_000:
        return True
    if "www.reddit.com/login" in lower and "comment" not in lower:
        return True
    return False


def _reddit_excerpt(payload: Any, *, fallback: str) -> tuple[str, str]:
    listing = payload if isinstance(payload, list) else [payload]
    title = ""
    parts: list[str] = []
    try:
        post = listing[0]["data"]["children"][0]["data"]
        title = post.get("title") or ""
        body = (post.get("selftext") or "").strip()
        sub = post.get("subreddit_name_prefixed") or post.get("subreddit") or ""
        if sub:
            parts.append(f"Subreddit: {sub}")
        if body:
            parts.append(body)
        if len(listing) > 1:
            comments = listing[1].get("data", {}).get("children", [])
            taken = 0
            for child in comments:
                if child.get("kind") != "t1":
                    continue
                body_c = (child.get("data") or {}).get("body") or ""
                if body_c.strip():
                    parts.append(f"Comment: {body_c.strip()}")
                    taken += 1
                if taken >= 5:
                    break
    except (KeyError, IndexError, TypeError):
        return title, fallback
    excerpt = "\n\n".join(parts).strip() or fallback
    return title, excerpt
