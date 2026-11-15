"""The minimal agent's hands: one pure function. It needs no resources, so there's no factory."""

from agentic_evals import tool


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text. Use it whenever a word count matters; don't count by eye."""
    return len(text.split())


TOOLS = [word_count]
