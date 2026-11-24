"""Command-line interface. Thin wrapper around :func:`memento_agent.research`."""

from __future__ import annotations

import memento_agent._warnings  # noqa: F401  — before urllib3/httpx via agent imports

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from memento_agent import __version__
from memento_agent.agent import research
from memento_agent.config import Settings
from memento_agent.errors import ConfigError, MementoError
from memento_agent.events import Event
from memento_agent.store import Store

app = typer.Typer(
    name="memento",
    help="Autonomous research agent: type a question, get a cited answer.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


def _on_event(event: Event) -> None:
    style = {
        "start": "bold cyan",
        "preflight": "cyan",
        "plan": "cyan",
        "planned": "cyan",
        "search": "blue",
        "fetch": "blue",
        "watch": "blue",
        "judge": "magenta",
        "judged": "magenta",
        "write": "green",
        "done": "bold green",
        "error": "bold red",
        "budget": "yellow",
    }.get(event.kind, "dim")
    console.print(f"[{style}]{event.kind}[/] {event.message}")


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        console.print(f"memento-agent {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command("research")
def research_cmd(
    question: list[str] = typer.Argument(..., help="Research question (quote it if it has spaces)."),
    model: Optional[str] = typer.Option(None, "--model", help="Gemini model for plan/judge/write."),
    search_provider: Optional[str] = typer.Option(
        None,
        "--search-provider",
        help="Search backend: tavily (default, free) or gemini (Google Search grounding; needs billing).",
    ),
    search_model: Optional[str] = typer.Option(
        None,
        "--search-model",
        help="Gemini model used with Google Search grounding when --search-provider gemini.",
    ),
    max_iterations: Optional[int] = typer.Option(None, "--max-iterations", min=1, max=8),
    max_searches: Optional[int] = typer.Option(None, "--max-searches", min=1, max=20),
    max_videos: Optional[int] = typer.Option(
        None,
        "--max-videos",
        min=0,
        max=10,
        help="Max YouTube videos to watch per run (default 2; Gemini free tier is 8h/day).",
    ),
    db: Optional[Path] = typer.Option(None, "--db", help="SQLite file for persisted runs."),
    as_json: bool = typer.Option(False, "--json", help="Print the result as JSON instead of markdown."),
) -> None:
    """Research a question autonomously and print a cited answer."""
    text = " ".join(question).strip()
    try:
        settings = Settings.from_env(
            model=model,
            search_model=search_model,
            search_provider=search_provider,
            db_path=db,
            max_iterations=max_iterations,
            max_searches=max_searches,
            max_videos=max_videos,
        )
        result = research(text, settings=settings, on_event=None if as_json else _on_event)
    except (ConfigError, MementoError, ValueError) as exc:
        err_console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    if as_json:
        console.print_json(data=result.to_dict())
        return
    console.print()
    console.print(Panel(Markdown(result.markdown), title=f"run {result.run_id}", border_style="green"))


@app.command("runs")
def list_runs(
    db: Optional[Path] = typer.Option(None, "--db"),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
) -> None:
    """List recent research runs from the local SQLite file."""
    settings = Settings.from_env(db_path=db, require_key=False)
    store = Store(settings.db_path)
    rows = store.list_runs(limit=limit)
    if not rows:
        console.print("No runs yet.")
        return
    for row in rows:
        q = str(row["question"]).replace("\n", " ")
        if len(q) > 90:
            q = q[:87] + "…"
        console.print(f"{row['id']}  {row['status']:<10}  {row['created_at']}  {q}")


@app.command("show")
def show_run(
    run_id: str = typer.Argument(..., help="Run id from `memento runs`."),
    db: Optional[Path] = typer.Option(None, "--db"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Print a stored run (markdown answer plus sources)."""
    settings = Settings.from_env(db_path=db, require_key=False)
    store = Store(settings.db_path)
    row = store.get_run(run_id)
    if row is None:
        err_console.print(f"[bold red]No run {run_id!r}.[/]")
        raise typer.Exit(code=1)
    if as_json:
        payload = dict(row)
        payload["sources"] = store.get_sources(run_id)
        payload["sub_queries"] = store.get_sub_queries(run_id)
        console.print(json.dumps(payload, indent=2, default=str))
        return
    markdown = row.get("answer_markdown") or f"# {row['question']}\n\n(no answer stored)\n"
    console.print(
        Panel(Markdown(str(markdown)), title=f"{row['id']} · {row['status']}", border_style="cyan")
    )


def main() -> None:
    app()
