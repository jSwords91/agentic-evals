"""The environment: the world an agent acts on, rebuilt from scratch for every trial.

To evaluate your own agent, subclass Environment and point a tasks file at it:

    class Workspace(Environment):
        def setup(self, workdir):  self.db = create_db(workdir / "workspace.db")
        def agent(self):           return build_agent(self.db)
        def query(self, sql):      ...  # read one value from the outcome, for `state` graders

    # tasks.yaml
    environment: my_package.env:Workspace

E8 [DE]: every trial gets a fresh environment, so no trial can leave state behind for the next.
  Shared state causes correlated failures, which make an agent look worse (or better) than it is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..agent import Agent


class Environment(ABC):
    @abstractmethod
    def setup(self, workdir: Path) -> None:
        """Create fresh state for one trial. `workdir` is an empty folder, deleted afterwards."""

    @abstractmethod
    def agent(self) -> Agent:
        """The agent under test, connected to this trial's state."""

    def query(self, query: str) -> Any:
        """One value read from the final state. `state` graders compare it with their target."""
        raise NotImplementedError(f"{type(self).__name__} has no query(), so it can't run state graders.")

    def snapshot(self) -> dict[str, Any]:
        """Anything worth keeping in the transcript, such as the system's own audit log."""
        return {}
