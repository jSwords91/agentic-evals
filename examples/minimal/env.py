"""The environment for the minimal agent.

Its only tool is a pure function: it reads its input and changes nothing. So there's no state
to create before a trial and nothing to read afterwards. Its tasks are graded on the reply
(`judge`) and on which tools it used (`tool_calls`), and `query` is never needed.
"""

from pathlib import Path

from agentic_evals import Agent
from agentic_evals.evals import Environment

from .agent import AGENT


class Stateless(Environment):
    def setup(self, workdir: Path) -> None:
        pass  # nothing to create

    def agent(self) -> Agent:
        return AGENT
