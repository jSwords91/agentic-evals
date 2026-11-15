"""Chat with Maya's assistant against a fresh copy of the workspace:

    uv run python -m examples.assistant
"""

import asyncio
import tempfile
from pathlib import Path

from agentic_evals import chat

from .agent import build_agent
from .data import create_db

with tempfile.TemporaryDirectory() as tmp:
    agent = build_agent(create_db(Path(tmp) / "workspace.db"))
    asyncio.run(chat(agent, "You're Maya Patel. Try: 'What's in my inbox?' or 'Book 30 minutes "
                            "with Sam and Lee on Wednesday.' Ctrl-D to quit."))
