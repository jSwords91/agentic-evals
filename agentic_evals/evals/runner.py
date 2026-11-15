"""The eval harness: run every task several times, each on a fresh environment, and grade it.

Most people will use it through the command line (see __main__.py). The pieces are:

    run_trial   one attempt at one task: fresh environment, agent, graders, transcript
    run_suite   many trials, a few at a time
    summarise   per-task pass@k, pass^k, cost and latency
    replay      a task's reference solution with no model, through the code graders (E9)
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
from mcp import MCPError

from ..agent import Agent, ToolCall
from .graders import Grade, Trial
from .metrics import pass_at_k, pass_hat_k
from .tasks import Suite, Task

# E13: what counts as infrastructure rather than the agent: API outages and rate limits, an MCP
#   server that failed to start or crashed, a missing binary. Deliberately narrow. See run_trial.
INFRA_ERRORS = (anthropic.APIError, MCPError, OSError)


@dataclass(frozen=True)
class Overrides:
    """Changes to the agent under test for one run: the experiments you'll actually make."""
    model: str | None = None
    effort: str | None = None
    no_screen: bool = False

    def apply(self, agent: Agent) -> Agent:
        changes: dict[str, Any] = {k: v for k, v in (("model", self.model), ("effort", self.effort)) if v}
        if self.no_screen:
            changes["screen"] = None
        return dataclasses.replace(agent, **changes)


async def run_trial(suite: Suite, task: Task, index: int, overrides: Overrides,
                    results_dir: Path, client: Any = None, judge_client: Any = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        env = suite.new_environment()
        env.setup(Path(tmp))
        agent = overrides.apply(env.agent())
        start = time.monotonic()
        infra_error = None
        # E13: an infrastructure error isn't an agent failure. The trial is left out of pass^k
        #   and reported separately, so an outage can't look like the agent getting worse.
        #   Only those errors are caught, and only around the steps that call out (the agent and
        #   the judge). A bug in the harness or a grader (a typo in a query, a KeyError) must
        #   stop the run: counted as infrastructure, it would quietly shrink n and leave pass^k
        #   looking healthy. `except*` because errors from inside the MCP client arrive in groups.
        try:
            async with agent.session(client) as session:
                reply = await session.send(task.input)
            seconds = time.monotonic() - start
            trial = Trial(task, reply.text, session.tool_log, env, judge_client)
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
        "cache_hit_rate": session.usage.cache_hit_rate, "seconds": seconds,
        "redactions": len(session.redactions), "transcript": str(path),
    }
    # E10 [DE]: the whole transcript, summary first, raw detail after. Reading transcripts is
    #   how you find out whether a failure is the agent's fault or the grader's.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        **summary,
        "input": task.input,
        "tool_log": [dataclasses.asdict(c) for c in session.tool_log],
        "redacted": session.redactions,
        "environment": snapshot,
        "messages": [_dump(m) for m in session.messages],
    }, indent=2, default=str, ensure_ascii=False))
    return summary


async def run_suite(suite: Suite, tasks: list[Task], trials: int, overrides: Overrides,
                    results_dir: Path, concurrency: int = 4, client: Any = None,
                    judge_client: Any = None) -> list[dict[str, Any]]:
    judge_client = judge_client or anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    async def limited(task: Task, i: int) -> dict[str, Any]:
        async with semaphore:
            return await run_trial(suite, task, i, overrides, results_dir, client, judge_client)

    return list(await asyncio.gather(*(limited(t, i) for t in tasks for i in range(trials))))


def summarise(tasks: list[Task], trials: list[dict[str, Any]], k: int) -> dict[str, dict[str, Any]]:
    summary = {}
    for task in tasks:
        runs = [t for t in trials if t["task"] == task.id]
        ok = [t for t in runs if "error" not in t]
        n, c = len(ok), sum(t["passed"] for t in ok)
        mean = {key: sum(t[key] for t in ok) / n if n else 0.0
                for key in ("score", "cost_usd", "seconds", "model_calls", "cache_hit_rate")}
        summary[task.id] = {
            "suite": task.suite, "n": n, "passed": c, "errors": len(runs) - n,
            "pass_at_k": pass_at_k(n, c, k) if n >= k else None,
            "pass_hat_k": pass_hat_k(n, c, k) if n >= k else None,
            **mean,
            "failed_graders": sorted({g for t in ok for g in t["failed"]}),
            "needs_review": sorted({g for t in ok for g in t["unknown"]}),
        }
    return summary


async def replay(suite: Suite, task: Task) -> tuple[list[Grade], list[ToolCall]]:
    """E9: run a task's reference solution with no model, then apply its code graders.

    Graders that read the reply are checked against the task's reference_reply, and skipped if
    it has none. Every grade should pass. If one doesn't, the task can't be passed as written or
    a grader is wrong, and you've found out for free, before any trial spends money. Returns the
    grades and the tool calls, so a caller can also spot a reference step that errored.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env = suite.new_environment()
        env.setup(Path(tmp))
        agent = dataclasses.replace(env.agent(), screen=None)
        async with agent.session() as session:
            for name, args in task.reference_solution:
                await session.call_tool(name, args)
        trial = Trial(task, task.reference_reply or "", session.tool_log, env, judge_client=None)
        checkable = [g for g in task.graders
                     if not g.uses_model and (task.reference_reply is not None or not g.reads_reply)]
        grades = [grade for spec in checkable for grade in await spec.grade(trial)]
        return grades, session.tool_log


def _leaves(error: BaseException) -> list[BaseException]:
    """The actual errors inside (possibly nested) exception groups."""
    if isinstance(error, BaseExceptionGroup):
        return [leaf for sub in error.exceptions for leaf in _leaves(sub)]
    return [error]


def _dump(message: dict[str, Any]) -> dict[str, Any]:
    content = message["content"]
    if not isinstance(content, str):
        content = [b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else b for b in content]
    return {"role": message["role"], "content": content}
