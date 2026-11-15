"""Ask the analyst agent questions about a fresh copy of the warehouse.

    uv run python -m examples.analyst
"""

import asyncio
import tempfile
from pathlib import Path

from agentic_evals import chat

from .agent import build_agent
from .warehouse import create_db

with tempfile.TemporaryDirectory() as tmp:
    agent = build_agent(create_db(Path(tmp) / "warehouse.db"))
    asyncio.run(chat(agent, "Brightleaf sales analyst. Try: 'What was our revenue in October?' Ctrl-D to quit."))
