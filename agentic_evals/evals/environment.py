"""The environment: the world an agent acts on, rebuilt from scratch for every trial.

An environment is only the world: fresh state for one trial, and a way to read the outcome
afterwards. It knows nothing about the agent under test. The agent's tools reach the trial's
world through a small function that takes the environment and returns the tools:

    class Warehouse(Environment):
        def setup(self, workdir):          # fresh state for one trial
            self.db = create_db(workdir / "warehouse.db")

        def query(self, sql):              # read one value from the outcome, for state graders
            ...

    def tools(env: Warehouse) -> list:     # the agent's hands, pointed at this trial's world
        return warehouse_tools(env.db)

Production builds the same tools with real resources: `warehouse_tools(production_db)`. So the
agent and its tools are identical in both places; only what the tools touch changes.

E8 [DE]: every trial gets a fresh environment, so no trial can leave state behind for the next.
  Shared state causes correlated failures, which make an agent look worse (or better) than it is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


class Environment(ABC):
    @abstractmethod
    def setup(self, workdir: Path) -> None:
        """Create fresh state for one trial. `workdir` is an empty folder, deleted afterwards.
        Keep whatever the tools will need (a database path, a fake client) as attributes."""

    def query(self, query: str) -> Any:
        """One value read from the final state. `state` graders compare it with their target."""
        raise NotImplementedError(f"{type(self).__name__} has no query(), so it can't run state graders.")

    def snapshot(self) -> dict[str, Any]:
        """Anything worth keeping in the transcript, such as the system's own audit log."""
        return {}


# Given a trial's environment, the tools the agent gets: `Tool` objects and MCP servers.
ToolFactory = Callable[[Any], Sequence[Any]]
