"""SQLite persistence for research runs. Stage 2/3 will query this same file."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from memento_agent.types import (
    Judgment,
    PlannedQuery,
    ResearchAnswer,
    SourceRecord,
    SourceType,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    source_types TEXT NOT NULL,
    status TEXT NOT NULL,
    model TEXT NOT NULL,
    search_model TEXT NOT NULL,
    answer_markdown TEXT,
    answer_json TEXT,
    error TEXT,
    iterations INTEGER NOT NULL DEFAULT 0,
    searches_used INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS sub_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    query TEXT NOT NULL,
    source_type TEXT NOT NULL,
    rationale TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    url TEXT NOT NULL,
    final_url TEXT,
    title TEXT,
    source_type TEXT NOT NULL,
    snippet TEXT,
    fetch_status TEXT NOT NULL,
    fetch_note TEXT,
    excerpt TEXT,
    query TEXT,
    iteration INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, id),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS judgments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_sub_queries_run ON sub_queries(run_id);
CREATE INDEX IF NOT EXISTS idx_sources_run ON sources(run_id);
CREATE INDEX IF NOT EXISTS idx_judgments_run ON judgments(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn = conn
        return conn

    def start_run(
        self,
        run_id: str,
        question: str,
        source_types: list[SourceType],
        *,
        model: str,
        search_model: str,
    ) -> None:
        self._connect().execute(
            """
            INSERT INTO runs (id, question, source_types, status, model, search_model, created_at)
            VALUES (?, ?, ?, 'running', ?, ?, ?)
            """,
            (
                run_id,
                question,
                json.dumps([t.value for t in source_types]),
                model,
                search_model,
                utcnow(),
            ),
        )
        self._connect().commit()

    def add_sub_queries(self, run_id: str, iteration: int, queries: list[PlannedQuery]) -> None:
        conn = self._connect()
        conn.executemany(
            """
            INSERT INTO sub_queries (run_id, iteration, query, source_type, rationale, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (run_id, iteration, q.query, q.source_type.value, q.rationale, utcnow())
                for q in queries
            ],
        )
        conn.commit()

    def add_source(self, run_id: str, source: SourceRecord) -> None:
        self._connect().execute(
            """
            INSERT OR REPLACE INTO sources (
                id, run_id, url, final_url, title, source_type, snippet,
                fetch_status, fetch_note, excerpt, query, iteration, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source.id,
                run_id,
                source.url,
                source.final_url,
                source.title,
                source.source_type.value,
                source.snippet,
                source.fetch_status.value,
                source.fetch_note,
                source.excerpt,
                source.query,
                source.iteration,
                utcnow(),
            ),
        )
        self._connect().commit()

    def add_judgment(self, run_id: str, iteration: int, judgment: Judgment) -> None:
        self._connect().execute(
            "INSERT INTO judgments (run_id, iteration, payload, created_at) VALUES (?, ?, ?, ?)",
            (run_id, iteration, judgment.model_dump_json(), utcnow()),
        )
        self._connect().commit()

    def complete_run(
        self,
        run_id: str,
        *,
        answer: ResearchAnswer,
        markdown: str,
        iterations: int,
        searches_used: int,
    ) -> None:
        self._connect().execute(
            """
            UPDATE runs
            SET status = 'completed',
                answer_markdown = ?,
                answer_json = ?,
                iterations = ?,
                searches_used = ?,
                completed_at = ?
            WHERE id = ?
            """,
            (markdown, answer.model_dump_json(), iterations, searches_used, utcnow(), run_id),
        )
        self._connect().commit()

    def fail_run(self, run_id: str, error: str, *, iterations: int = 0, searches_used: int = 0) -> None:
        self._connect().execute(
            """
            UPDATE runs
            SET status = 'failed', error = ?, iterations = ?, searches_used = ?, completed_at = ?
            WHERE id = ?
            """,
            (error, iterations, searches_used, utcnow(), run_id),
        )
        self._connect().commit()

    def list_runs(self, limit: int = 20) -> list[dict[str, object]]:
        rows = self._connect().execute(
            """
            SELECT id, question, status, created_at, completed_at, iterations, searches_used
            FROM runs ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, object] | None:
        row = self._connect().execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_sources(self, run_id: str) -> list[dict[str, object]]:
        rows = self._connect().execute(
            "SELECT * FROM sources WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def get_sub_queries(self, run_id: str) -> list[dict[str, object]]:
        rows = self._connect().execute(
            "SELECT * FROM sub_queries WHERE run_id = ? ORDER BY iteration, id", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]
