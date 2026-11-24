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
    Judgment,
    ResearchAnswer,
    ResearchPlan,
    SourceRecord,
    SourceType,
    VideoWatchNotes,
)


class GeminiLLM:
    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client
        self._model = settings.model
        self._search_model = settings.effective_search_model

    @property
    def model(self) -> str:
        return self._model

    def _client_or_create(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._settings.api_key)
        return self._client

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
    ) -> Judgment:
        raw = self._generate_json(
            JUDGE_SYSTEM,
            judge_user_prompt(question, source_types, sources, iteration, max_iterations),
            Judgment,
        )
        return Judgment.model_validate(raw)

    def write(
        self,
        question: str,
        sources: list[SourceRecord],
        source_types: list[SourceType],
        judgment: Judgment,
    ) -> ResearchAnswer:
        citable = [s for s in sources if s.citable]
        raw = self._generate_json(
            WRITE_SYSTEM,
            write_user_prompt(question, source_types, citable, judgment),
            ResearchAnswer,
        )
        return ResearchAnswer.model_validate(raw)

    def watch_video(self, url: str, *, title: str = "") -> VideoWatchNotes:
        from google.genai import types

        prompt = VIDEO_WATCH_PROMPT.format(title=title or url)

        def config():
            return types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VideoWatchNotes,
                temperature=0.1,
            )

        if hasattr(types.Part, "from_uri"):
            video_part = types.Part.from_uri(file_uri=url, mime_type="video/mp4")
        else:
            video_part = types.Part(
                file_data=types.FileData(file_uri=url, mime_type="video/mp4")
            )
        contents = [video_part, prompt]
        # Video watches are capped per run; 1–2 attempts then the caller falls back to captions.
        response = self._generate(contents=contents, config_factory=config, attempts=2)
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
    ) -> Any:
        client = self._client_or_create()
        candidates = models or self._candidate_models()
        last_err: Exception | None = None
        for index, model in enumerate(candidates):
            remaining = len(candidates) - index - 1
            tries = attempts if attempts is not None else (2 if remaining else 4)

            def _call(model: str = model) -> Any:
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config_factory(),
                )

            try:
                response = call_with_retry(_call, max_attempts=tries, grounded=grounded)
                # Do not let a successful grounded-search model overwrite plan/judge/write.
                if models is None:
                    self._model = model
                return response
            except ModelUnavailableError as exc:
                last_err = exc
                continue
            except RateLimitError as exc:
                if remaining and ("503" in str(exc) or "server error" in str(exc).lower()):
                    last_err = exc
                    continue
                raise
        raise last_err or LLMError("All configured Gemini models failed.")


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
