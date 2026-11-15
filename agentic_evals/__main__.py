"""The command line.

    python -m agentic_evals check  TASKS                 validate tasks, replay reference solutions (free)
    python -m agentic_evals run    TASKS [options]       run the evals
    python -m agentic_evals screen CASES [options]       the injection screen's component eval

Run it from your project's root, so the environment named in the tasks file can be imported.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import anthropic

from .evals import load_suite
from .evals.metrics import regression_gate
from .evals.report import print_comparison, print_failures, print_table
from .evals.runner import Overrides, replay, run_suite, summarise
from .evals.screening import load_cases, print_report, run_cases, score
from .screen import Screen


async def check(args: argparse.Namespace) -> int:
    suite = load_suite(args.tasks)  # raises on any malformed task
    print(f"{len(suite.tasks)} tasks in {args.tasks} are valid. Replaying reference solutions:\n")
    failures = 0
    for task in suite.tasks:
        grades, calls = await replay(suite, task)
        bad = [g for g in grades if not g.passed]
        failures += bool(bad)
        print(f"  {'ok' if not bad else 'FAIL':6}{task.id}")
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


async def run(args: argparse.Namespace) -> int:
    suite = load_suite(args.tasks)
    k = args.k or args.trials
    if unknown := sorted(set(args.task) - {t.id for t in suite.tasks}):
        raise SystemExit(f"No task called {', '.join(unknown)} in {args.tasks}.")
    tasks = [t for t in suite.tasks
             if args.suite in ("all", t.suite) and (not args.task or t.id in args.task)]
    overrides = Overrides(model=args.model, effort=args.effort, no_screen=args.no_screen,
                          without_tools=tuple(args.without_tool))
    await preflight()
    trials = await run_suite(suite, tasks, args.trials, overrides, args.results, args.concurrency)
    summary = summarise(tasks, trials, k)

    changed = {key: value for key, value in vars(overrides).items() if value}
    print(f"{args.tasks}: {len(tasks)} tasks x {args.trials} trials, k={k}, "
          f"overrides: {changed or 'none'}")
    print_table(summary, k)
    if args.compare:
        print_comparison(summary, json.loads(args.compare.read_text())["tasks"])
    print_failures(trials)
    args.results.mkdir(parents=True, exist_ok=True)
    out = args.results / "latest.json"
    out.write_text(json.dumps({"overrides": vars(overrides), "k": k, "tasks": summary,
                               "trials": trials}, indent=2, default=str))
    print(f"\nWrote {out} and transcripts in {args.results / 'transcripts'}")

    if args.gate is None:
        return 0
    mean, passed = regression_gate(
        [r["pass_hat_k"] for r in summary.values() if r["suite"] == "regression"], args.gate)
    print("Regression gate: FAIL, no regression tasks ran" if mean is None
          else f"Regression gate: mean pass^{k} {mean:.2f} (need {args.gate}): "
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
    p.add_argument("--model", help="override the agent's model")
    p.add_argument("--effort", help="override the agent's effort")
    p.add_argument("--no-screen", action="store_true", help="turn off the injection screen")
    p.add_argument("--without-tool", action="append", default=[], metavar="NAME",
                   help="leave a tool out (repeatable): does it earn its place?")
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
