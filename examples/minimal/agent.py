"""The smallest useful agent: one tool and a one-line prompt.

    uv run python -m examples.minimal
"""

from agentic_evals import Agent, tool


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text. Use it whenever a word count matters; don't count by eye."""
    return len(text.split())


AGENT = Agent(system="You are a concise writing assistant.", tools=[word_count])
