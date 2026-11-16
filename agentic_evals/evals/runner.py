"""The eval harness: run every task several times, each in a fresh environment, and grade it.

    results = await evaluate(suite, agent=ANALYST, tools=tools, trials=3)
    results.table()

The agent and tools under test are arguments, not part of the suite. So the same tasks can be
run against a first draft of an agent and its next version, or against two models, and the
results compared task by task. The pieces are:

    evaluate    many trials, a few at a time, summarised into Results
    run_trial   one attempt at one task: fresh environment, session, graders, transcript
    replay      a task's reference solution with no model, through the code graders (E9)
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import tempfile
import time
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Literal

import anthropic
from mcp import MCPError

from ..agent import Agent
from ..hands import Hands
from ..harness import Harness
from ..session import Session
from ..tools import ToolCall
from .environment import ToolFactory
from .graders import Grade, Trial
from .metrics import pass_at_k, pass_hat_k
from .results import Results
from .tasks import Suite, Task

# E13: what counts as infrastructure rather than the agent: API outages and rate limits, an MCP
#   server that failed to start or crashed, a missing binary. Deliberately narrow. See run_trial.
INFRA_ERRORS = (anthropic.APIError, MCPError, OSError)


async def evaluate(suite: Suite, *, agent: Agent | None = None, tools: ToolFactory | None = None,
                   trials: int = 3, k: int | None = None,
                   include: Literal["regression", "capability", "all"] = "all",
                   task_ids: Collection[str] = (), results_dir: str | Path = "results",
                   concurrency: int = 4, client: Any = None, judge_client: Any = None) -> Results:
    """Run the suite's tasks against `agent` with `tools` (by default, the suite's own).

    `include` and `task_ids` pick the tasks. `client` is the API client for the agent and
    `judge_client` the one for judges; leave both out to use ANTHROPIC_API_KEY.
    """
    agent = agent or suite.agent
    if agent is None:
        raise ValueError("No agent to test: pass agent=, or name one in the suite.")
    tools = tools or suite.tools or (lambda env: [])
    if unknown := sorted(set(task_ids) - {t.id for t in suite.tasks}):
        raise ValueError(f"No task called {', '.join(unknown)} in this suite.")
    tasks = [t for t in suite.tasks
             if include in ("all", t.suite) and (not task_ids or t.id in task_ids)]
    k = k or trials
    results_dir = Path(results_dir)
    judge_client = judge_client or anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    async def limited(task: Task, i: int) -> dict[str, Any]:
        async with semaphore:
            return await run_trial(suite, task, i, agent, tools, results_dir, client, judge_client)

    runs = list(await asyncio.gather(*(limited(t, i) for t in tasks for i in range(trials))))
    return Results(agent={"model": agent.model, "effort": agent.effort, "screen": agent.screen is not None},
                   k=k, tasks=summarise(tasks, runs, k), trials=runs)


async def run_trial(suite: Suite, task: Task, index: int, agent: Agent, tools: ToolFactory,
                    results_dir: Path, client: Any = None, judge_client: Any = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        env = suite.environment()
        env.setup(Path(tmp))
        start = time.monotonic()
        infra_error = None
        # E13: an infrastructure error isn't an agent failure. The trial is left out of pass^k
        #   and reported separately, so an outage can't look like the agent getting worse.
        #   Only those errors are caught, and only around the steps that call out (the agent and
        #   the judge). A bug in the harness or a grader (a typo in a query, a KeyError) must
        #   stop the run: counted as infrastructure, it would quietly shrink n and leave pass^k
        #   looking healthy. `except*` because errors from inside the MCP client arrive in groups.
        session = Session()  # E8: a fresh log per trial, like the environment
        try:
            async with Hands(tools(env)) as hands:
                reply = await Harness(agent, hands, session, client).send(task.input)
            seconds = time.monotonic() - start
            trial = Trial(task, reply.text, session.tool_calls, env, judge_client)
            judged = await asyncio.gather(*(g.grade(trial) for g in task.graders if g.uses_model))
        except* INFRA_ERRORS as group:
            infra_error = group
        if infra_error is not None:  # no transcript for these
            return {"task": task.id, "trial": index,
                    "error": "; ".join(repr(e) for e in _leaves(infra_error))}
        grades = [g for spec in task.graders if not spec.uses_model for g in await spec.grade(trial)]
        grades += [g for batch in judged for g in batch]
        snapshot = env.snapshot()  # read before the environment's folder is deleted

    path = results_dir / "transcripts" / f"{task.id}-{index}.json"
    summary = {
        "task": task.id, "trial": index, "suite": task.suite,
        "model": agent.model, "effort": agent.effort, "screen": agent.screen is not None,
        "passed": all(g.passed for g in grades),
        # E11 [DE]: partial credit too, so progress shows before a task flips to passing.
        "score": sum(g.passed for g in grades) / len(grades),
        "failed": [g.name for g in grades if not g.passed],
        "unknown": [g.name for g in grades if g.unknown],  # for a person to review
        "grades": [dataclasses.asdict(g) for g in grades],
        "reply": reply.text, "stop_reason": reply.stop_reason,
        # E10/K2: the numbers that decide whether a change is worth making.
        "model_calls": session.usage.calls, "cost_usd": session.cost_usd,
        # [TOOLS]: many failed tool calls point at a tool's description or its error messages.
        "tool_errors": sum(c.is_error for c in session.tool_calls),
        "cache_hit_rate": session.usage.cache_hit_rate, "seconds": seconds,
        "redactions": len(session.redactions), "transcript": str(path),
    }
    # E10 [DE]: the whole transcript, summary first, raw detail after. Reading transcripts is
    #   how you find out whether a failure is the agent's fault or the grader's. The raw detail
    #   is the session itself: the same log production keeps, so a trial reads like a real
    #   conversation. The tool log and redactions are read from it, for quick reading.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        **summary,
        "input": task.input,
        "tool_log": [dataclasses.asdict(c) for c in session.tool_calls],
        "redacted": session.redactions,
        "environment": snapshot,
        "session": [e.model_dump(mode="json") for e in session],
    }, indent=2, default=str, ensure_ascii=False))
    return summary


def summarise(tasks: Sequence[Task], trials: list[dict[str, Any]], k: int) -> dict[str, dict[str, Any]]:
    summary = {}
    for task in tasks:
        runs = [t for t in trials if t["task"] == task.id]
        ok = [t for t in runs if "error" not in t]
        n, c = len(ok), sum(t["passed"] for t in ok)
        mean = {key: sum(t[key] for t in ok) / n if n else 0.0
                for key in ("score", "cost_usd", "seconds", "model_calls", "tool_errors", "cache_hit_rate")}
        summary[task.id] = {
            "suite": task.suite, "n": n, "passed": c, "errors": len(runs) - n,
            "pass_at_k": pass_at_k(n, c, k) if n >= k else None,
            "pass_hat_k": pass_hat_k(n, c, k) if n >= k else None,
            **mean,
            "failed_graders": sorted({g for t in ok for g in t["failed"]}),
            "needs_review": sorted({g for t in ok for g in t["unknown"]}),
        }
    return summary


async def replay(suite: Suite, task: Task, tools: ToolFactory | None = None) -> tuple[list[Grade], list[ToolCall]]:
    """E9: run a task's reference solution on the hands alone, then apply its code graders.

    Graders that read the reply are checked against the task's reference_reply, and skipped if
    it has none. Every grade should pass. If one doesn't, the task can't be passed as written or
    a grader is wrong, and you've found out for free, before any trial spends money. Returns the
    grades and the tool calls, so a caller can also spot a reference step that errored.
    """
    # No brain at all: the hands don't need one, which is what keeping them apart buys (A4).
    tools = tools or suite.tools or (lambda env: [])
    with tempfile.TemporaryDirectory() as tmp:
        env = suite.environment()
        env.setup(Path(tmp))
        async with Hands(tools(env)) as hands:
            tool_calls = [await hands.execute(name, args) for name, args in task.reference_solution]
        trial = Trial(task, task.reference_reply or "", tool_calls, env, judge_client=None)
        checkable = [g for g in task.graders
                     if not g.uses_model and (task.reference_reply is not None or not g.reads_reply)]
        grades = [grade for spec in checkable for grade in await spec.grade(trial)]
        return grades, tool_calls


def _leaves(error: BaseException) -> list[BaseException]:
    """The actual errors inside (possibly nested) exception groups."""
    if isinstance(error, BaseExceptionGroup):
        return [leaf for sub in error.exceptions for leaf in _leaves(sub)]
    return [error]

