"""Ask the analyst agent questions about a fresh copy of the warehouse.

    uv run python -m examples.analyst
"""

import asyncio
import tempfile
from pathlib import Path

from agentic_evals import chat

from .agent import ANALYST
from .tools import warehouse_tools
from .warehouse import create_db

with tempfile.TemporaryDirectory() as tmp:
    answers = []  # a real app would show each recorded answer, with its SQL, beside the reply
    tools = warehouse_tools(create_db(Path(tmp) / "warehouse.db"), answers)
    asyncio.run(chat(ANALYST, tools, "Brightleaf sales analyst. Try: 'What was our revenue in October?' "
                                     "Ctrl-D to quit."))
