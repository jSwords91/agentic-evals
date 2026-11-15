import asyncio

from agentic_evals import chat

from .agent import WRITER
from .tools import TOOLS

asyncio.run(chat(WRITER, TOOLS, "A writing assistant with a word_count tool. Ctrl-D to quit."))
