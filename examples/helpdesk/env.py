"""The eval world for the helpdesk agent.

Each trial gets a fresh ticket database and a status page with fixed answers. `tools` points the
agent's tools at them. `query` runs SQL against the database, so `state` graders can check the
priority and team the agent set.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from agentic_evals import Tool
from agentic_evals.evals import Environment

from .db import create_db
from .status import FixedStatusPage
from .tools import ticket_tools

STATUS = {
    "vpn": "Degraded. Known incident INC-311: VPN connections dropping. Engineers are working on a fix.",
    "email": "Operational. No known incidents.",
    "wifi": "Operational. No known incidents.",
    "payroll": "Operational. No known incidents.",
}


class Helpdesk(Environment):
    def setup(self, workdir: Path) -> None:
        self.db = create_db(workdir / "tickets.db")
        self.status = FixedStatusPage(STATUS)  # a fake, so every trial sees the same incidents

    def query(self, query: str) -> Any:
        with closing(sqlite3.connect(self.db)) as con:
            return con.execute(query).fetchone()[0]

    def snapshot(self) -> dict[str, Any]:
        with closing(sqlite3.connect(self.db)) as con:
            return {"tickets": con.execute("SELECT id, status, priority, team FROM tickets").fetchall(),
                    "replies": con.execute("SELECT ticket_id, body FROM replies").fetchall()}


def tools(env: Helpdesk) -> list[Tool]:
    """The helpdesk's tools, pointed at this trial's database and status page."""
    return ticket_tools(env.db, env.status)
