"""The smallest useful agent: one tool, one question.

    uv run python examples/minimal.py
"""

import asyncio

from agentic_evals import Agent, tool


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


async def main() -> None:
    agent = Agent(system="You are a concise writing assistant.", tools=[word_count])
    async with agent.session() as session:
        reply = await session.send("How many words are in 'the quick brown fox jumps'?")
        print(reply.text)
        print(f"Tools used: {[c.name for c in session.tool_log]}, cost ${session.cost_usd:.4f}")


asyncio.run(main())
