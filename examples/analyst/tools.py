"""The analyst's hands: read-only tools for the sales warehouse.

`warehouse_tools(db, answers)` builds them for a database and a list that recorded answers go
to. Production passes the real warehouse and a list the app shows to the team; an eval passes a
fresh copy of the warehouse and a list the graders read. The tools don't know which.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agentic_evals import Tool, ToolError, tool

from .warehouse import COLUMNS, TABLES

MAX_ROWS = 50        # rows shown to the model; also stated in the tool descriptions
MAX_SECONDS = 5      # a query that runs longer is stopped, so one bad join can't hang a trial

Table = Literal["customers", "products", "orders", "order_items"]


@dataclass(frozen=True)
class RecordedAnswer:
    sql: str
    columns: list[str]
    rows: list[tuple]


def connect(db: Path) -> sqlite3.Connection:
    """S3 [CONTAIN]: read-only, enforced by SQLite, not requested in the prompt. An UPDATE or
    DELETE fails whatever the model writes."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    deadline = time.monotonic() + MAX_SECONDS
    con.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)  # non-zero aborts
    return con


def run(db: Path, sql: str) -> tuple[list[str], list[tuple]]:
    """Column names and rows. SQL errors go back to the model so it can fix its query (A5)."""
    try:
        with closing(connect(db)) as con:
            cursor = con.execute(sql)
            return [d[0] for d in cursor.description or []], cursor.fetchall()
    except sqlite3.Error as e:
        raise ToolError(f"SQL error: {e}. describe_table shows each table's columns.") from None


def format_rows(columns: list[str], rows: list[tuple]) -> str:
    """T4 [TOOLS]: compact, capped output. When rows are cut, say so and say how to narrow the
    query, so the model doesn't reason from a partial result."""
    lines = [" | ".join(columns)] + [" | ".join("NULL" if v is None else str(v) for v in r) for r in rows[:MAX_ROWS]]
    more = (f"\n({len(rows)} rows; showing the first {MAX_ROWS}. Aggregate, filter or add a LIMIT to see "
            "what you need.)" if len(rows) > MAX_ROWS else f"\n({len(rows)} rows)")
    return "\n".join(lines) + more


def warehouse_tools(db: Path, answers: list[RecordedAnswer]) -> list[Tool]:
    """Tools that read the warehouse at `db`. Recorded answers are appended to `answers`: the
    agent's deliverable, which the app shows to the team, and which an eval grades."""

    @tool
    def list_tables() -> str:
        """The tables in the sales warehouse, what one row of each is, and how many rows it has."""
        return "\n".join(f"{name}: {about} ({run(db, f'SELECT COUNT(*) FROM {name}')[1][0][0]} rows)"
                         for name, about in TABLES.items())

    @tool
    def describe_table(table: Table) -> str:
        """A table's columns, with their types and what they mean, and three example rows.
        Read this before querying a table: it says how tables join and what coded values mean."""
        _, info = run(db, f"PRAGMA table_info({table})")  # the live schema, so it's never out of date
        columns = [f"  {c[1]} ({c[2]}): {COLUMNS[table].get(c[1], 'no description')}" for c in info]
        return f"{table}: {TABLES[table]}\n" + "\n".join(columns) + "\n\nExample rows:\n" + format_rows(
            *run(db, f"SELECT * FROM {table} LIMIT 3"))

    @tool
    def run_sql(sql: str) -> str:
        """Run a read-only SQLite query on the warehouse and see the result (at most 50 rows).
        Use it to explore and to check your working."""
        return format_rows(*run(db, sql))

    @tool
    def record_answer(sql: str) -> str:
        """Record the query that answers the question. Its result is the official answer: the team
        sees it and can rerun it. Select only what answers the question: a single figure, or for a
        breakdown one row per group with its label and figure. Round as the definitions say."""
        columns, rows = run(db, sql)
        if not rows or len(rows) > MAX_ROWS:
            fix = "Check the filters." if not rows else "Aggregate, or select one row per group."
            raise ToolError(f"An answer needs between 1 and {MAX_ROWS} rows; this query returned {len(rows)}. {fix}")
        answers.append(RecordedAnswer(sql, columns, rows))
        return "Recorded. The answer is:\n" + format_rows(columns, rows)

    return [list_tables, describe_table, run_sql, record_answer]
