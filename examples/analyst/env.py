"""The eval world for the analyst agent, and the graders written for it.

Each trial gets a fresh copy of the warehouse, and an empty list for the answers the agent
records; `tools` points the agent's tools at both. The agent can't change the warehouse (its
connection is read-only), so the copy matters less for isolation than for trust: the graders
run reference queries on exactly the data the agent saw.

E14: the graders here are the heart of this example's evals. All of them grade the recorded
answer, the agent's deliverable, never which tools it called:

  answer               the recorded answer holds the result of a reference query, whatever
                       else it adds. The task stores the query, not the number, so changing the
                       seed data never means editing expected values by hand, and the query
                       documents the right answer.
  reply_states_answer  the reply gives the right figure too, so the number a person reads is
                       the number the query produced.
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

from agentic_evals import Tool
from agentic_evals.evals import Environment, Grade, Grader, Trial, register

from .tools import RecordedAnswer, warehouse_tools
from .warehouse import create_db


class Warehouse(Environment):
    def setup(self, workdir: Path) -> None:
        self.db = create_db(workdir / "warehouse.db")
        self.answers: list[RecordedAnswer] = []

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



def tools(env: Warehouse) -> list[Tool]:
    """The analyst's tools, pointed at this trial's warehouse and answer list."""
    return warehouse_tools(env.db, env.answers)

def contains(got: tuple, expected: tuple, tolerance: float) -> bool:
    """Whether a recorded row holds every label and every figure of an expected row.

    Column names and order don't matter, and extra columns are allowed: an answer that adds
    context, such as the order count beside an average or the two inputs to a rate, is still
    right. Text is compared without case, numbers within `tolerance`.
    """
    labels = [v.strip().lower() for v in got if isinstance(v, str)]
    figures = [v for v in got if isinstance(v, int | float)]
    for value in expected:
        pool = labels if isinstance(value, str) else figures
        match = next((v for v in pool if (v == value.strip().lower() if isinstance(value, str)
                                          else abs(v - value) <= tolerance)), None)
        if match is None:
            return False
        pool.remove(match)  # each recorded value can stand for one expected value only
    return True


def matches(got: list[tuple], expected: list[tuple], tolerance: float, ordered: bool) -> bool:
    """One recorded row per expected row: a missing or extra group fails, and so does the wrong
    order when the order is part of the answer."""
    if len(got) != len(expected):
        return False
    if ordered:
        return all(contains(g, e, tolerance) for g, e in zip(got, expected, strict=True))
    # Pair every expected row with its own recorded row. Taking the first recorded row that fits
    #   isn't enough once extra columns are allowed: (1, 2) fits both (1,) and (1, 2), and could
    #   be taken by the wrong one. So this finds a full pairing if there is one (Kuhn's algorithm).
    fits = [[contains(g, e, tolerance) for g in got] for e in expected]
    paired: dict[int, int] = {}  # recorded row -> expected row

    def pair(e: int, seen: set[int]) -> bool:
        for g in range(len(got)):
            if fits[e][g] and g not in seen:
                seen.add(g)
                if g not in paired or pair(paired[g], seen):
                    paired[g] = e
                    return True
        return False

    return all(pair(e, set()) for e in range(len(expected)))


@register
class Answer(Grader):
    """The recorded answer holds the reference query's result, row for row.

    It grades whether the answer is right, not its layout: no output format is imposed on the
    agent, so extra columns pass (see `contains`). The cost is that an answer recording the right
    figure beside a wrong one also passes; where that kind of hedging is plausible, add
    `reply_states_answer` so the figure a person reads is checked too.
    """
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
        passed = matches(got, expected, self.tolerance, self.ordered)
        return [Grade(self.name, passed, f"got {got[:5]}, expected {expected[:5]}")]


NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


@register
class ReplyStatesAnswer(Grader):
    """The reply states the right figure, at whatever precision it uses ("£4,160.10", "4160.1"
    and "£4,160" all match 4160.1). Paired with `answer`, which checks the recorded answer holds
    the same figure, it makes the number a person reads the number the query produced.

    `reference_sql` returns the one right figure. It's usually the `answer` grader's query; a
    YAML anchor saves writing it twice.
    """
    type: Literal["reply_states_answer"] = "reply_states_answer"
    name: str
    reference_sql: str
    reads_reply: ClassVar[bool] = True

    async def grade(self, trial: Trial) -> list[Grade]:
        assert isinstance(trial.env, Warehouse)
        value = Decimal(str(trial.env.query(self.reference_sql)))
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
