"""A data analyst agent: answers the commercial team's questions from the sales warehouse.

    uv run python -m examples.analyst

The team has to be able to trust its numbers, so the design starts from what could make a
number wrong, and puts each defence where it holds most firmly:

  - Wrong tables or joins. The agent reads the live schema and a description of every column
    (describe_table) before it writes SQL.
  - Wrong definition (counting refunds, including test accounts). The agreed definitions are a
    skill, loaded before any calculation.
  - A number the model made up, or misremembered between query and reply. The answer is
    recorded as SQL (record_answer), and its value is computed by code from that SQL. A person
    can rerun it, and the evals check it against a reference query.
  - Changing the data. The connection is opened read-only, so no prompt can change it (S3).
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Literal

from agentic_evals import Agent, ToolError, tool

from .warehouse import COLUMNS, TABLES

HERE = Path(__file__).parent
MAX_ROWS = 50        # rows shown to the model; also stated in the tool descriptions
MAX_SECONDS = 5      # a query that runs longer is stopped, so one bad join can't hang a trial

Table = Literal["customers", "products", "orders", "order_items"]

SYSTEM = """You answer questions about sales for the commercial team at Brightleaf, a UK online shop selling houseplants, pots and plant care. It's Monday 16 November 2026.

Every figure you give must come from a query you ran on the warehouse. Look at the tables you need before you write SQL, and load the metric definitions before you calculate anything.

When you have the answer, record the query that produces it with record_answer. Then reply in a few sentences: the figure exactly as recorded, the period it covers, and what it counts.

If the data can't answer the question, say so and say what's missing. Never estimate a figure. You have read-only access to the warehouse."""


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
        raise ToolError(f"SQL error: {e}") from None


def table(columns: list[str], rows: list[tuple]) -> str:
    """T4 [TOOLS]: compact, capped output. Say how many rows there were, so a cut-off is visible."""
    lines = [" | ".join(columns)] + [" | ".join("NULL" if v is None else str(v) for v in r) for r in rows[:MAX_ROWS]]
    more = f"\n({len(rows)} rows; showing the first {MAX_ROWS})" if len(rows) > MAX_ROWS else f"\n({len(rows)} rows)"
    return "\n".join(lines) + more


def build_agent(db: Path) -> Agent:
    @tool
    def list_tables() -> str:
        """The tables in the sales warehouse, what one row of each is, and how many rows it has."""
        return "\n".join(f"{name}: {about} ({run(db, f'SELECT COUNT(*) FROM {name}')[1][0][0]} rows)"
                         for name, about in TABLES.items())

    @tool
    def describe_table(name: Table) -> str:
        """A table's columns, with their types and what they mean, and three example rows.
        Read this before querying a table: it says how tables join and what coded values mean."""
        _, info = run(db, f"PRAGMA table_info({name})")  # the live schema, so it's never out of date
        columns = [f"  {c[1]} ({c[2]}): {COLUMNS[name].get(c[1], 'no description')}" for c in info]
        return f"{name}: {TABLES[name]}\n" + "\n".join(columns) + "\n\nExample rows:\n" + table(
            *run(db, f"SELECT * FROM {name} LIMIT 3"))

    @tool
    def run_sql(sql: str) -> str:
        """Run a read-only SQLite query on the warehouse and see the result (at most 50 rows).
        Use it to explore and to check your working."""
        return table(*run(db, sql))

    @tool
    def record_answer(sql: str) -> str:
        """Record the query that answers the question. Its result is the official answer: the team
        sees it and can rerun it. Select only what answers the question: a single figure, or for a
        breakdown one row per group with its label and figure. Round as the definitions say."""
        columns, rows = run(db, sql)
        if not rows or len(rows) > MAX_ROWS:
            raise ToolError(f"An answer needs between 1 and {MAX_ROWS} rows; this query returned {len(rows)}.")
        return "Recorded. The answer is:\n" + table(columns, rows)

    return Agent(system=SYSTEM, tools=[list_tables, describe_table, run_sql, record_answer],
                 skills=HERE / "skills")
