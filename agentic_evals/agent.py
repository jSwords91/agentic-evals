"""The agent: the brain's configuration. What Claude is told and what it's allowed to do.

    agent = Agent(system="You help with ...")                          # the brain
    session = Session()                                                 # the log
    async with Hands([convert_time, stdio("python", "-m", "my_server")]) as hands:   # the hands
        reply = await Harness(agent, hands, session).send("What's on Thursday?")

A4 [MA]: Anthropic's Managed Agents design splits an agent into three parts with narrow
  interfaces, so each can fail or be replaced without the others:
  - the brain, "Claude and its harness": `Agent` (here) is what the brain is told and allowed
    to do, and `Harness` (harness.py) is the loop that does it.
  - the hands, "sandboxes and tools that perform actions": `Hands` (hands.py).
  - the session, "the append-only log of everything that happened": `Session` (session.py).
  An `Agent` holds no tools, no resources and no state, so it's a plain constant you can share,
  compare and change with `dataclasses.replace(agent, model=...)`.
  Managed Agents runs the hands in sandboxes, behind a credential vault, on other machines. This
  reference keeps the same interfaces in one process; nothing at this scale needs the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .screen import Screen
from .usage import check_priced


@dataclass(frozen=True)
class Agent:
    system: str
    skills: str | Path | None = None     # a folder of */SKILL.md files; see skills.py
    screen: Screen | None = None         # the injection screen; see screen.py
    # K1 [docs]: model and effort are configuration, so trying another model is a flag on an
    #   eval run, not a code change. Opus 5.5 defaults to medium effort. It's set explicitly so
    #   that switching models doesn't silently change it.
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_tokens: int = 16000  # per response; a safe ceiling for non-streaming calls
    # A2 [HD][CONTAIN]: hard stops enforced by the harness, not requested of the model.
    max_turns: int = 20
    max_cost_usd: float = 1.00
    # E10 [docs]: Opus 5.5 always thinks, but by default returns the thinking empty. "summarized"
    #   puts a readable summary of its reasoning into every transcript, at no extra cost.
    thinking_display: str = "summarized"

    def __post_init__(self) -> None:
        check_priced(self.model)
        if self.screen:
            check_priced(self.screen.model)
