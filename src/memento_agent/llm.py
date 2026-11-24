"""Gemini client for planning, judging, writing, video watch, and optional grounding."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from memento_agent.config import Settings
from memento_agent.errors import LLMError, ModelUnavailableError, RateLimitError
from memento_agent.prompts import (
    JUDGE_SYSTEM,
    PLAN_SYSTEM,
    VIDEO_WATCH_PROMPT,
    WRITE_SYSTEM,
    judge_user_prompt,
    plan_user_prompt,
    write_user_prompt,
)
from memento_agent.retry import call_with_retry
from memento_agent.types import (
    ExtractedEntity,
    Judgment,
    ResearchAnswer,
    ResearchPlan,
    SourceRecord,
    SourceType,
    VideoWatchNotes,
)


class GeminiLLM:
    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        on_fallback: Any | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._injected_client = client is not None
        self._watch_client = None
        self._model = settings.model
        self._search_model = settings.effective_search_model
        self.on_fallback = on_fallback

    @property
    def model(self) -> str:
        return self._model

    def _client_or_create(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._settings.api_key)
        return self._client

    def _watch_client_or_create(self) -> Any:
        # Tests inject a dummy client. Live watches need a dedicated 600s v1beta client;
        # a non-streaming default client disconnected around 270s in a 24 Nov 2026 run.
        if self._injected_client:
            return self._client
        if self._watch_client is None:
            from google import genai
            from google.genai import types

            timeout_ms = int(self._settings.watch_timeout_s * 1000)
            try:
                http = types.HttpOptions(timeout=timeout_ms, api_version="v1beta")
            except TypeError:
                try:
                    http = types.HttpOptions(timeout=timeout_ms)
                except TypeError:
                    http = types.HttpOptions(api_version="v1beta")
            self._watch_client = genai.Client(api_key=self._settings.api_key, http_options=http)
        return self._watch_client

    def _candidate_models(self) -> list[str]:
        models = [self._model]
        for name in self._settings.fallback_models:
            if name and name not in models:
                models.append(name)
        return models

    def generate_search(self, prompt: str) -> Any:
        from google.genai import types

        def config():
            return types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            )

        return self._generate(
            contents=prompt,
            config_factory=config,
            models=[self._search_model],
            grounded=True,
        )

    def probe(self) -> str:
        """Tiny generate to prove the model works. 404/503 fall back. list() is not trusted."""
        from google.genai import types

        def config():
            return types.GenerateContentConfig(temperature=0)

        self._generate(contents="Reply with the single word pong.", config_factory=config)
        return self._model

    def plan(self, question: str, source_types: list[SourceType]) -> ResearchPlan:
        raw = self._generate_json(
            PLAN_SYSTEM,
            plan_user_prompt(question, source_types),
            ResearchPlan,
        )
        plan = ResearchPlan.model_validate(raw)
        if not plan.source_types:
            plan.source_types = list(source_types)
        if not plan.sub_queries:
            raise LLMError("The planner returned no sub-queries.")
        return plan

    def judge(
        self,
        question: str,
        sources: list[SourceRecord],
        source_types: list[SourceType],
        *,
        iteration: int,
        max_iterations: int,
        entities: list[ExtractedEntity] | None = None,
        excerpt_limit: int | None = None,
    ) -> Judgment:
        limit = excerpt_limit or self._settings.judge_excerpt_chars
        try:
            return self._judge_once(
                question,
                sources,
                source_types,
                iteration=iteration,
                max_iterations=max_iterations,
                entities=entities,
                excerpt_limit=limit,
            )
        except RateLimitError:
            return self._judge_once(
                question,
                sources,
                source_types,
                iteration=iteration,
                max_iterations=max_iterations,
                entities=entities,
                excerpt_limit=min(600, limit),
            )

    def _judge_once(
        self,
        question: str,
        sources: list[SourceRecord],
        source_types: list[SourceType],
        *,
        iteration: int,
        max_iterations: int,
        entities: list[ExtractedEntity] | None,
        excerpt_limit: int,
    ) -> Judgment:
        raw = self._generate_json(
            JUDGE_SYSTEM,
            judge_user_prompt(
                question,
                source_types,
                sources,
                iteration,
                max_iterations,
                entities=entities,
                excerpt_limit=excerpt_limit,
            ),
            Judgment,
        )
        return Judgment.model_validate(raw)

    def write(
        self,
        question: str,
        sources: list[SourceRecord],
        source_types: list[SourceType],
        judgment: Judgment,
        entities: list[ExtractedEntity] | None = None,
    ) -> ResearchAnswer:
        citable = [s for s in sources if s.citable]
        raw = self._generate_json(
            WRITE_SYSTEM,
            write_user_prompt(question, source_types, citable, judgment, entities=entities),
            ResearchAnswer,
        )
        return ResearchAnswer.model_validate(raw)

    def watch_video(self, url: str, *, title: str = "") -> VideoWatchNotes:
        from google.genai import types

        prompt = VIDEO_WATCH_PROMPT.format(title=title or url)
        video_part = _video_part(types, url)

        def config():
            kwargs: dict[str, Any] = {
                "response_mime_type": "application/json",
                "response_schema": VideoWatchNotes,
                "temperature": 0.1,
            }
            media = getattr(types, "MediaResolution", None)
            if media is not None:
                kwargs["media_resolution"] = getattr(media, "MEDIA_RESOLUTION_LOW", media)
            return types.GenerateContentConfig(**kwargs)

        contents = [video_part, prompt]
        response = self._generate(
            contents=contents,
            config_factory=config,
            attempts=3,
            stream=True,
            client=self._watch_client_or_create(),
        )
        text = getattr(response, "text", None) or ""
        if not text.strip():
            raise LLMError("Gemini returned an empty video-watch response.")
        try:
            return VideoWatchNotes.model_validate(json.loads(_strip_fences(text)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMError(f"Gemini video-watch JSON was invalid: {exc}") from exc

    def _generate_json(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> dict[str, Any]:
        from google.genai import types

        def config():
            return types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.2,
            )

        response = self._generate(contents=user, config_factory=config)
        text = getattr(response, "text", None) or ""
        if not text.strip():
            raise LLMError("Gemini returned an empty JSON response.")
        try:
            return json.loads(_strip_fences(text))
        except json.JSONDecodeError as exc:
            try:
                return schema.model_validate_json(_strip_fences(text)).model_dump()
            except ValidationError:
                raise LLMError(f"Gemini returned invalid JSON: {exc}") from exc

    def _generate(
        self,
        *,
        contents: Any,
        config_factory: Any,
        models: list[str] | None = None,
        attempts: int | None = None,
        grounded: bool = False,
        stream: bool = False,
        client: Any | None = None,
    ) -> Any:
        client = client or self._client_or_create()
        candidates = models or self._candidate_models()
        last_err: Exception | None = None
        for index, model in enumerate(candidates):
            remaining = len(candidates) - index - 1
            tries = attempts if attempts is not None else (2 if remaining else 4)
            nxt = candidates[index + 1] if remaining else None

            def _call(model: str = model) -> Any:
                if stream and hasattr(client.models, "generate_content_stream"):
                    return _collect_stream(
                        client.models.generate_content_stream(
                            model=model,
                            contents=contents,
                            config=config_factory(),
                        )
                    )
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config_factory(),
                )

            try:
                response = call_with_retry(_call, max_attempts=tries, grounded=grounded)
                if models is None:
                    self._model = model
                return response
            except ModelUnavailableError as exc:
                last_err = exc
                if nxt:
                    self._emit_fallback(model, nxt, exc)
                continue
            except RateLimitError as exc:
                if remaining:
                    last_err = exc
                    if nxt:
                        self._emit_fallback(model, nxt, exc)
                    continue
                raise
        raise last_err or LLMError("All configured Gemini models failed.")

    def _emit_fallback(self, current: str, nxt: str, exc: BaseException) -> None:
        if not self.on_fallback:
            return
        short = str(exc).strip().split("\n", 1)[0][:180]
        self.on_fallback(f"{current} failed ({type(exc).__name__}: {short}); trying {nxt}")


def _video_part(types: Any, url: str) -> Any:
    """YouTube FileData part with STATIC frame extraction and LOW media resolution."""
    processing = getattr(getattr(types, "MediaProcessing", None), "STATIC", "STATIC")
    part = None
    if hasattr(types.Part, "from_uri"):
        for extra in (
            {"media_resolution": "MEDIA_RESOLUTION_LOW"},
            {},
        ):
            try:
                part = types.Part.from_uri(file_uri=url, mime_type="video/mp4", **extra)
                break
            except TypeError:
                continue
    if part is None:
        file_kwargs: dict[str, Any] = {"file_uri": url, "mime_type": "video/mp4"}
        try:
            data = types.FileData(**file_kwargs)
        except TypeError:
            data = types.FileData(file_uri=url)
        try:
            part = types.Part(file_data=data, media_processing=processing)
        except TypeError:
            part = types.Part(file_data=data)
    if part is not None and getattr(part, "media_processing", None) is None:
        try:
            part = part.model_copy(update={"media_processing": processing})
        except Exception:
            try:
                part.media_processing = processing
            except Exception:
                pass
    return part


def _collect_stream(chunks: Any) -> Any:
    from types import SimpleNamespace

    parts: list[str] = []
    last = None
    for chunk in chunks:
        last = chunk
        text = getattr(chunk, "text", None)
        if text:
            parts.append(text)
    combined = "".join(parts)
    if last is not None and not hasattr(last, "text"):
        return last
    return SimpleNamespace(text=combined, candidates=getattr(last, "candidates", None))


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
