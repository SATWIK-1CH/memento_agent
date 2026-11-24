"""Gemini client for planning, judging, writing, and grounded search."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from memento_agent.config import Settings
from memento_agent.errors import LLMError
from memento_agent.prompts import (
    JUDGE_SYSTEM,
    PLAN_SYSTEM,
    WRITE_SYSTEM,
    judge_user_prompt,
    plan_user_prompt,
    write_user_prompt,
)
from memento_agent.retry import call_with_retry
from memento_agent.types import Judgment, ResearchAnswer, ResearchPlan, SourceRecord, SourceType


class GeminiLLM:
    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client
        self._model = settings.model
        self._search_model = settings.effective_search_model

    def _client_or_create(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._settings.api_key)
        return self._client

    def generate_search(self, prompt: str) -> Any:
        from google.genai import types

        client = self._client_or_create()
        return client.models.generate_content(
            model=self._search_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )

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

    def _generate_json(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> dict[str, Any]:
        from google.genai import types

        client = self._client_or_create()

        def _call() -> Any:
            return client.models.generate_content(
                model=self._model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.2,
                ),
            )

        response = call_with_retry(_call)
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
