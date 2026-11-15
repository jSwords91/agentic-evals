"""The command line.

    python -m agentic_evals check  TASKS                 validate tasks, replay reference solutions (free)
    python -m agentic_evals run    TASKS [options]       run the evals
    python -m agentic_evals screen CASES [options]       the injection screen's component eval

Run it from your project's root, so the names in the tasks file can be imported.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
from pathlib import Path

import anthropic

from .evals import evaluate, load_suite
from .evals.runner import replay
from .evals.screening import load_cases, print_report, run_cases, score
from .evals.tasks import import_object
from .screen import Screen


async def check(args: argparse.Namespace) -> int:
    suite = load_suite(args.tasks)  # raises on any malformed task
    print(f"{len(suite.tasks)} tasks in {args.tasks} are valid. Replaying reference solutions:\n")
    failures = 0
    for task in suite.tasks:
        grades, calls = await replay(suite, task)
        bad = [g for g in grades if not g.passed]
        failures += bool(bad)
        unchecked = any(g.reads_reply for g in task.graders) and task.reference_reply is None
        print(f"  {'ok' if not bad else 'FAIL':6}{task.id}"
              + ("  (reply graders not checked: add a reference_reply)" if unchecked else ""))
        for g in bad:
            print(f"        x {g.name}: {g.detail}")
        for c in calls:
            if c.is_error:  # sometimes intended (a refusal is the point), often a typo
                print(f"        ! reference step {c.name} returned an error: {c.output[:120]}")
    return 1 if failures else 0


async def preflight() -> None:
    """Fail in a second, with a plain message, rather than after starting every trial."""
    try:
        await anthropic.AsyncAnthropic().models.list(limit=1)  # free, and needs valid credentials
    except (TypeError, anthropic.AuthenticationError) as e:  # TypeError: no credentials at all
        raise SystemExit(f"Can't call the Claude API: {e}\nSet ANTHROPIC_API_KEY and try again.") from None


def named(target: str) -> object:
    """An --agent or --tools argument, with a plain message if it can't be imported."""
    try:
        return import_object(target)
    except (ImportError, AttributeError, ValueError) as e:
        raise SystemExit(f"Can't import {target}: {e}") from None


async def run(args: argparse.Namespace) -> int:
    suite = load_suite(args.tasks)
    agent = named(args.agent) if args.agent else suite.agent
    if agent is None:
        raise SystemExit(f"{args.tasks} names no agent: pass --agent package.module:NAME.")
    # K1: the experiments you'll actually run are a different model, effort or screen, on the
    #   same agent. Each is a new Agent; the original is unchanged.
    changes = {key: value for key, value in (("model", args.model), ("effort", args.effort)) if value}
    if args.no_screen:
        changes["screen"] = None
    agent = dataclasses.replace(agent, **changes)
    tools = named(args.tools) if args.tools else None
    await preflight()
    try:
        results = await evaluate(suite, agent=agent, tools=tools, trials=args.trials, k=args.k,
                                 include=args.suite, task_ids=args.task, results_dir=args.results,
                                 concurrency=args.concurrency)
    except ValueError as e:  # an unknown --task, say
        raise SystemExit(str(e)) from None

    results.table()
    if args.compare:
        results.compare(args.compare)
    results.failures()
    out = results.save(args.results / "latest.json")
    print(f"\nWrote {out} and transcripts in {args.results / 'transcripts'}")

    if args.gate is None:
        return 0
    mean, passed = results.gate(args.gate)
    print("Regression gate: FAIL, no regression tasks ran" if mean is None
          else f"Regression gate: mean pass^{results.k} {mean:.2f} (need {args.gate}): "
               + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


async def screen(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    await preflight()
    config = Screen(model=args.model, effort=args.effort)
    result = score(cases, await run_cases(cases, config, args.trials, args.concurrency))
    return 0 if print_report(result, config, args.trials, args.min_catch, args.min_keep) else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m agentic_evals")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("check", help="validate tasks and replay reference solutions, no API calls")
    p.add_argument("tasks", type=Path)

    p = commands.add_parser("run", help="run the evals")
    p.add_argument("tasks", type=Path)
    p.add_argument("--suite", choices=["regression", "capability", "all"], default="all")
    p.add_argument("--task", action="append", default=[], help="run only this task id (repeatable)")
    p.add_argument("--trials", type=int, default=3, help="trials per task (n)")
    p.add_argument("--k", type=int, help="k for pass@k and pass^k (default: --trials)")
    p.add_argument("--agent", metavar="MODULE:NAME", help="the agent to test (default: the tasks file's)")
    p.add_argument("--tools", metavar="MODULE:NAME", help="the tools to give it (default: the tasks file's)")
    p.add_argument("--model", help="override the agent's model")
    p.add_argument("--effort", help="override the agent's effort")
    p.add_argument("--no-screen", action="store_true", help="turn off the injection screen")
    p.add_argument("--gate", type=float, help="exit 1 unless mean regression pass^k reaches this")
    p.add_argument("--compare", type=Path, help="an earlier results JSON to compare against")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--results", type=Path, default=Path("results"))

    p = commands.add_parser("screen", help="the injection screen's component eval")
    p.add_argument("cases", type=Path)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--model", default=Screen.model)
    p.add_argument("--effort", default=Screen.effort)
    p.add_argument("--min-catch", type=float, default=1.0, help="share of attacks caught every time")
    p.add_argument("--min-keep", type=float, default=0.9, help="share of harmless text kept every time")
    p.add_argument("--concurrency", type=int, default=8)

    args = parser.parse_args()
    return asyncio.run({"check": check, "run": run, "screen": screen}[args.command](args))


if __name__ == "__main__":
    sys.exit(main())
