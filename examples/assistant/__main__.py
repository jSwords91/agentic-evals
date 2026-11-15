"""Chat with Maya's assistant against a fresh copy of the workspace:

    uv run python -m examples.assistant
"""

import asyncio
import tempfile
from pathlib import Path

from agentic_evals import chat

from .agent import ASSISTANT
from .data import create_db
from .tools import workspace_tools

with tempfile.TemporaryDirectory() as tmp:
    tools = workspace_tools(create_db(Path(tmp) / "workspace.db"))
    asyncio.run(chat(ASSISTANT, tools, "You're Maya Patel. Try: 'What's in my inbox?' or 'Book 30 minutes "
                            "with Sam and Lee on Wednesday.' Ctrl-D to quit."))
