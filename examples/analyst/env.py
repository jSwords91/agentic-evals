"""The eval environment for the analyst agent, and the graders written for it.

Each trial gets a fresh copy of the warehouse, and an empty list for the answers the agent
records. The agent can't change the warehouse (its connection is read-only), so the copy matters
less for isolation than for trust: the graders run reference queries on exactly the data the
agent saw.

E14: the graders here are the heart of this example's evals. All of them grade the recorded
answer, the agent's deliverable, never which tools it called:

  answer               the recorded answer matches a reference query. The task stores the
                       query, not the number, so changing the seed data never means editing
                       expected values by hand, and the query documents the right answer.
  reply_states_answer  the reply gives the same figure as the recorded answer, so the number
                       a person reads is the number the query produced.
  no_answer            nothing was recorded: for a question the data can't answer, no figure
                       should reach the team.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import asdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, ClassVar, Literal

from agentic_evals import Agent
from agentic_evals.evals import Environment, Grade, Grader, Trial, register

from .agent import RecordedAnswer, build_agent
from .warehouse import create_db


class Warehouse(Environment):
    def setup(self, workdir: Path) -> None:
        self.db = create_db(workdir / "warehouse.db")
        self.answers: list[RecordedAnswer] = []

    def agent(self) -> Agent:
        return build_agent(self.db, self.answers)

    def query(self, query: str) -> Any:
        return self.rows(query)[0][0]

    def rows(self, query: str) -> list[tuple]:
        with closing(sqlite3.connect(self.db)) as con:
            return con.execute(query).fetchall()

    def snapshot(self) -> dict[str, Any]:
        return {"answers": [asdict(a) for a in self.answers]}

    def answer(self) -> RecordedAnswer | None:
        """The latest recorded answer: the one the team would see."""
        return self.answers[-1] if self.answers else None


def normalise(rows: list[tuple], ordered: bool) -> list[tuple]:
    """Rows made comparable. Within a row, text labels come first (lower-cased), then numbers,
    so column names and column order don't matter. Rows are sorted unless their order is part of
    the answer."""
    def row(r: tuple) -> tuple:
        labels = sorted(v.strip().lower() for v in r if isinstance(v, str))
        figures = sorted(v for v in r if not isinstance(v, str))
        return (*labels, *figures)
    rows = [row(r) for r in rows]
    return rows if ordered else sorted(rows, key=str)


def same(got: list[tuple], expected: list[tuple], tolerance: float) -> bool:
    if len(got) != len(expected):
        return False
    for g_row, e_row in zip(got, expected, strict=True):
        if len(g_row) != len(e_row):
            return False
        for g, e in zip(g_row, e_row, strict=True):
            numbers = isinstance(g, int | float) and isinstance(e, int | float)
            if (abs(g - e) > tolerance) if numbers else g != e:
                return False
    return True


@register
class Answer(Grader):
    type: Literal["answer"] = "answer"
    name: str
    reference_sql: str
    tolerance: float = 0.01   # absolute; money is reported to the penny
    ordered: bool = False     # true when the order is part of the answer, as in a top three

    async def grade(self, trial: Trial) -> list[Grade]:
        assert isinstance(trial.env, Warehouse)
        answer = trial.env.answer()
        if answer is None:
            return [Grade(self.name, False, "no answer was recorded")]
        got = answer.rows
        expected = trial.env.rows(self.reference_sql)  # a broken reference is a grader bug: let it raise
        passed = same(normalise(got, self.ordered), normalise(expected, self.ordered), self.tolerance)
        return [Grade(self.name, passed, f"got {got[:5]}, expected {expected[:5]}")]


NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


@register
class ReplyStatesAnswer(Grader):
    """For an answer that's a single figure: the reply contains that figure, at whatever
    precision the reply uses ("£4,160.10", "4160.1" and "£4,160" all match 4160.1)."""
    type: Literal["reply_states_answer"] = "reply_states_answer"
    name: str
    reads_reply: ClassVar[bool] = True

    async def grade(self, trial: Trial) -> list[Grade]:
        assert isinstance(trial.env, Warehouse)
        answer = trial.env.answer()
        rows = answer.rows if answer else []
        if len(rows) != 1 or len(rows[0]) != 1 or not isinstance(rows[0][0], int | float):
            return [Grade(self.name, False, f"the recorded answer isn't a single figure: {rows[:3]}")]
        value = Decimal(str(rows[0][0]))
        for text in NUMBER.findall(trial.reply):
            stated = Decimal(text.replace(",", ""))
            places = Decimal(1).scaleb(stated.as_tuple().exponent)  # the precision the reply used
            if value.quantize(places, rounding=ROUND_HALF_UP) == stated:
                return [Grade(self.name, True, f"states {text}")]
        return [Grade(self.name, False, f"no number in the reply matches {value}")]


@register
class NoAnswer(Grader):
    """For a question the data can't answer: no figure was recorded, so none reached the team."""
    type: Literal["no_answer"] = "no_answer"
    name: str

    async def grade(self, trial: Trial) -> list[Grade]:
        assert isinstance(trial.env, Warehouse)
        answer = trial.env.answer()
        return [Grade(self.name, answer is None, f"recorded: {answer.rows[:3]}" if answer else "")]
