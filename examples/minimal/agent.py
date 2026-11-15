"""The minimal agent's brain: a one-line prompt.

    uv run python -m examples.minimal
"""

from agentic_evals import Agent

WRITER = Agent(system="You are a concise writing assistant.")
