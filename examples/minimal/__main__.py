import asyncio

from agentic_evals import chat

from .agent import AGENT

asyncio.run(chat(AGENT, "A writing assistant with a word_count tool. Ctrl-D to quit."))
