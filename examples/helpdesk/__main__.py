"""Chat with the helpdesk agent against a fresh ticket database and the fixed status page.

    uv run python -m examples.helpdesk
"""

import asyncio
import tempfile
from pathlib import Path

from agentic_evals import chat

from .agent import HELPDESK
from .db import create_db
from .env import STATUS
from .status import FixedStatusPage
from .tools import ticket_tools

with tempfile.TemporaryDirectory() as tmp:
    tools = ticket_tools(create_db(Path(tmp) / "tickets.db"), FixedStatusPage(STATUS))
    asyncio.run(chat(HELPDESK, tools, "IT helpdesk. Try: 'Can you triage ticket 101?' Tickets 101-104 exist. "
                                      "Ctrl-D to quit."))
