"""Every example's tasks, and the helpdesk example's agent and environment."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agentic_evals.evals import load_suite
from agentic_evals.evals.runner import Overrides, replay, run_trial
from examples.helpdesk.env import Helpdesk
from tests.conftest import ScriptedModel, response, text, tool_use

EXAMPLES = Path(__file__).parent.parent / "examples"
SUITES = {path.parent.name: load_suite(path) for path in sorted(EXAMPLES.glob("*/tasks.yaml"))}
TASKS = [(name, task) for name, suite in SUITES.items() for task in suite.tasks]


def test_every_example_has_a_tasks_file():
    assert set(SUITES) == {"assistant", "helpdesk", "minimal"}


@pytest.mark.parametrize("example, task", TASKS, ids=[f"{n}/{t.id}" for n, t in TASKS])
def test_reference_solution_passes_every_code_grader(example, task):
    """E9: each task can be passed, and its code graders accept a correct solution."""
    grades, calls = asyncio.run(replay(SUITES[example], task))
    assert all(g.passed for g in grades), grades
    assert not any(c.is_error for c in calls), calls


# ---------------------------------------------------------------- the helpdesk

def helpdesk_trial(tmp_path: Path, task_id: str, *steps) -> dict:
    """One trial of a helpdesk task with a scripted model that makes `steps`, then replies."""
    suite = SUITES["helpdesk"]
    task = next(t for t in suite.tasks if t.id == task_id)
    calls = [response("tool_use", tool_use(str(i), name, **args)) for i, (name, args) in enumerate(steps)]
    model = ScriptedModel(*calls, response("end_turn", text("Done.")))
    return asyncio.run(run_trial(suite, task, 0, Overrides(), tmp_path, model, ScriptedModel()))


def test_a_correct_triage_passes(tmp_path):
    task = next(t for t in SUITES["helpdesk"].tasks if t.id == "known_incident_is_normal")
    assert helpdesk_trial(tmp_path, task.id, *task.reference_solution)["passed"]


def test_ignoring_the_status_page_fails(tmp_path):
    # Treating the VPN ticket as a fresh outage is the mistake the status check exists to prevent.
    result = helpdesk_trial(tmp_path, "known_incident_is_normal",
                            ("triage_ticket", {"ticket_id": 101, "priority": "urgent", "team": "network"}))
    assert "network_normal" in result["failed"]


def test_each_trial_gets_its_own_database(tmp_path):
    first, second = Helpdesk(), Helpdesk()
    for env, folder in ((first, tmp_path / "a"), (second, tmp_path / "b")):
        folder.mkdir()
        env.setup(folder)
    triage = next(t for t in first.agent().tools if t.name == "triage_ticket")
    asyncio.run(triage.run({"ticket_id": 102, "priority": "low", "team": "hardware"}))
    assert first.query("SELECT COUNT(*) FROM tickets WHERE status = 'triaged'") == 1
    assert second.query("SELECT COUNT(*) FROM tickets WHERE status = 'triaged'") == 0
