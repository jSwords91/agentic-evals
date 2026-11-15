"""Tasks: one input, the graders that decide pass or fail, and a reference solution.

A tasks file names the environment the tasks run in, then lists the tasks:

    environment: examples.assistant.env:Workspace
    tasks:
      - id: question_changes_nothing
        suite: regression
        input: What's on my calendar on Thursday?
        graders:
          - {type: state, name: no_events_added, query: "SELECT COUNT(*) FROM events", expect: "== 9"}
          - type: judge
            dimensions:
              mentions_site_visit: The reply mentions the site visit at 10:00.
        reference_solution:
          - [list_events, {date_from: "2026-11-19", date_to: "2026-11-19"}]

E1 [DE]: the vocabulary used throughout:
  task        one input plus its graders and reference solution
  trial       one attempt at a task; outputs vary between runs, so each task gets several
  transcript  the full record of a trial: messages, tool calls, cost (results/transcripts/)
  outcome     the final state of the environment, not what the agent says it did
  grader      code or a model that scores one aspect of a trial
  harness     the eval harness runs the trials; the agent harness (agent.py) is under test
E6 [DE]: two suites. Regression tasks should pass nearly every time and gate every change.
  Capability tasks start with a low pass rate, giving you something to climb. They're reported
  but never block a merge, and graduate to regression once they pass reliably.
E9 [DE]: a reference solution is a list of tool calls that passes every code grader. It proves
  the task can be solved and the graders are configured correctly. `python -m agentic_evals
  check` replays them with no model, for free.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from .environment import Environment
from .graders import Grader, parse_grader


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    suite: Literal["regression", "capability"]
    input: str
    graders: list[Grader] = Field(min_length=1)
    reference_solution: list[tuple[str, dict[str, Any]]] = []

    @field_validator("graders", mode="before")
    @classmethod
    def _parse_graders(cls, specs: Any) -> Any:
        return [parse_grader(s) for s in specs] if isinstance(specs, list) else specs


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    environment: str  # "module:ClassName", importable from the working directory
    tasks: list[Task] = Field(min_length=1)
    _environment_class: type[Environment] = PrivateAttr()

    @model_validator(mode="after")
    def _unique_ids(self) -> Suite:
        ids = [t.id for t in self.tasks]
        if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
            raise ValueError(f"duplicate task ids: {', '.join(duplicates)}")
        return self

    def new_environment(self) -> Environment:
        return self._environment_class()


def load_suite(path: str | Path) -> Suite:
    data = yaml.safe_load(Path(path).read_text())
    # The environment's module is imported before any task is read, so custom graders defined
    # there are registered in time.
    environment_class = import_environment(data.get("environment", ""))
    suite = Suite.model_validate(data)
    suite._environment_class = environment_class
    return suite


def import_environment(target: str) -> type[Environment]:
    module, _, name = target.partition(":")
    if not (module and name):
        raise ValueError(f"environment must look like 'package.module:ClassName', got {target!r}")
    cls = getattr(importlib.import_module(module), name)
    if not (isinstance(cls, type) and issubclass(cls, Environment)):
        raise ValueError(f"{target} is not an Environment subclass")
    return cls
