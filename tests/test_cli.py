from __future__ import annotations

from typer.testing import CliRunner

from memento_agent.cli import app
from memento_agent.types import ResearchAnswer, ResearchResult, SourceType

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "memento-agent" in result.stdout


def test_research_uses_injected_agent(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_research(question, **kwargs):
        captured["question"] = question
        return ResearchResult(
            run_id="deadbeef",
            question=question,
            source_types=[SourceType.WEB],
            answer=ResearchAnswer(summary="ok", sections=[], limitations=[]),
            markdown="# ok\n",
            sources=[],
        )

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr("memento_agent.cli.research", fake_research)
    result = runner.invoke(
        app,
        ["research", "Best hotels in Agonda", "--db", str(tmp_path / "x.db"), "--json"],
    )
    assert result.exit_code == 0, result.output
    assert captured["question"] == "Best hotels in Agonda"
    assert "deadbeef" in result.stdout


def test_research_without_key_fails(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = runner.invoke(app, ["research", "Agonda"])
    assert result.exit_code == 1
    assert "GEMINI_API_KEY" in result.output


def test_research_without_tavily_key_fails(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("MEMENTO_SEARCH_PROVIDER", raising=False)
    result = runner.invoke(app, ["research", "Agonda"])
    assert result.exit_code == 1
    assert "TAVILY_API_KEY" in result.output
    assert "billing" in result.output.lower() or "grounding" in result.output.lower()


def test_runs_without_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = runner.invoke(app, ["runs", "--db", str(tmp_path / "empty.db")])
    assert result.exit_code == 0
    assert "No runs" in result.stdout
