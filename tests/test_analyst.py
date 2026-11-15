"""The analyst example: its read-only warehouse, and the graders that decide whether a number is right."""

from __future__ import annotations

import asyncio

import pytest

from agentic_evals import ToolError
from agentic_evals.agent import ToolCall
from agentic_evals.evals import Task, Trial
from examples.analyst import agent as analyst
from examples.analyst.env import Answer, ReplyStatesAnswer, Warehouse

OCTOBER = "o.ordered_at >= '2026-10-01' AND o.ordered_at < '2026-11-01'"
JOINS = ("FROM orders o JOIN customers c ON c.id = o.customer_id "
         "JOIN order_items i ON i.order_id = o.id")
REVENUE = (f"SELECT ROUND(SUM(i.quantity * i.unit_price), 2) {JOINS} "
           f"WHERE o.status = 'completed' AND c.is_test = 0 AND {OCTOBER}")
NAIVE = f"SELECT ROUND(SUM(i.quantity * i.unit_price), 2) {JOINS} WHERE {OCTOBER}"
TASK = Task.model_validate({"id": "t", "suite": "regression", "input": "?",
                            "graders": [{"type": "answer", "name": "a", "reference_sql": REVENUE}]})


@pytest.fixture
def warehouse(tmp_path) -> Warehouse:
    env = Warehouse()
    env.setup(tmp_path)
    return env


def tool(warehouse: Warehouse, name: str):
    return next(t for t in warehouse.agent().tools if t.name == name)


def trial(warehouse: Warehouse, recorded_sql: str | None, reply: str = "") -> Trial:
    log = [] if recorded_sql is None else [ToolCall("record_answer", {"sql": recorded_sql}, "Recorded.", False)]
    return Trial(TASK, reply, log, warehouse, None)


def grade(grader, t: Trial):
    [result] = asyncio.run(grader.grade(t))
    return result


# ---------------------------------------------------------------- the warehouse (S3, T4)

def test_the_warehouse_cannot_be_changed(warehouse):
    with pytest.raises(ToolError, match="readonly"):
        asyncio.run(tool(warehouse, "run_sql").run({"sql": "DELETE FROM orders WHERE status = 'cancelled'"}))
    assert warehouse.query("SELECT COUNT(*) FROM orders WHERE status = 'cancelled'") == 44


def test_a_runaway_query_is_stopped(warehouse, monkeypatch):
    monkeypatch.setattr(analyst, "MAX_SECONDS", 0)
    with pytest.raises(ToolError, match="interrupted"):
        asyncio.run(tool(warehouse, "run_sql").run(
            {"sql": "SELECT COUNT(*) FROM order_items a, order_items b, order_items c"}))


def test_the_agent_sees_column_descriptions_and_example_rows(warehouse):
    text = asyncio.run(tool(warehouse, "describe_table").run({"name": "customers"}))
    assert "is_test (INTEGER): 1 for internal test accounts" in text and "Example rows" in text


def test_sql_errors_go_back_to_the_model(warehouse):
    with pytest.raises(ToolError, match="no such column"):
        asyncio.run(tool(warehouse, "run_sql").run({"sql": "SELECT revenue FROM orders"}))


# ---------------------------------------------------------------- the answer grader (E2, E14)

def test_the_right_query_passes_however_it_is_written(warehouse):
    rewritten = ("SELECT ROUND(SUM(quantity * unit_price), 2) AS revenue FROM order_items WHERE order_id IN "
                 "(SELECT o.id FROM orders o JOIN customers c ON c.id = o.customer_id "
                 "WHERE o.status = 'completed' AND NOT c.is_test AND strftime('%Y-%m', o.ordered_at) = '2026-10')")
    assert grade(Answer(name="a", reference_sql=REVENUE), trial(warehouse, rewritten)).passed


def test_a_plausible_wrong_query_fails(warehouse):
    result = grade(Answer(name="a", reference_sql=REVENUE), trial(warehouse, NAIVE))
    assert not result.passed and "5903.45" in result.detail and "4160.1" in result.detail


def test_column_order_and_names_do_not_matter_but_row_order_can(warehouse):
    by_region = f"SELECT c.region, SUM(i.quantity) {JOINS} WHERE {OCTOBER} GROUP BY c.region"
    swapped = (f"SELECT SUM(i.quantity) AS units, c.region AS r {JOINS} "
               f"WHERE {OCTOBER} GROUP BY c.region ORDER BY r DESC")
    assert grade(Answer(name="a", reference_sql=by_region), trial(warehouse, swapped)).passed
    assert not grade(Answer(name="a", reference_sql=by_region, ordered=True), trial(warehouse, swapped)).passed


def test_no_recorded_answer_fails(warehouse):
    assert grade(Answer(name="a", reference_sql=REVENUE), trial(warehouse, None)).detail == "no answer was recorded"


@pytest.mark.parametrize("reply, passed", [
    ("Revenue in October was £4,160.10.", True),
    ("About £4,160 in October.", True),        # rounded, but to the figure
    ("It was 4160.1.", True),
    ("Revenue was £4,200.", False),            # a different figure
    ("Revenue was £5,903.45.", False),
])
def test_the_reply_must_state_the_recorded_figure(warehouse, reply, passed):
    assert grade(ReplyStatesAnswer(name="r"), trial(warehouse, REVENUE, reply)).passed is passed
