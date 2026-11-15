"""Chat with the helpdesk agent against a fresh ticket database and the fixed status page.

    uv run python -m examples.helpdesk
"""

import asyncio
import tempfile
from pathlib import Path

from agentic_evals import chat

from .agent import build_agent
from .db import create_db
from .env import STATUS
from .status import FixedStatusPage

with tempfile.TemporaryDirectory() as tmp:
    agent = build_agent(create_db(Path(tmp) / "tickets.db"), FixedStatusPage(STATUS))
    asyncio.run(chat(agent, "IT helpdesk. Try: 'Triage ticket 101.' Tickets 101-104 exist. Ctrl-D to quit."))
