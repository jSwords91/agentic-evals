"""The eval environment for the helpdesk agent.

Each trial gets a fresh ticket database and a status page with fixed answers. `query` runs
SQL against the database, so `state` graders can check the priority and team the agent set.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from agentic_evals import Agent
from agentic_evals.evals import Environment

from .agent import build_agent
from .db import create_db
from .status import FixedStatusPage

STATUS = {
    "vpn": "Degraded. Known incident INC-311: VPN connections dropping. Engineers are working on a fix.",
    "email": "Operational. No known incidents.",
    "wifi": "Operational. No known incidents.",
    "payroll": "Operational. No known incidents.",
}


class Helpdesk(Environment):
    def setup(self, workdir: Path) -> None:
        self.db = create_db(workdir / "tickets.db")

    def agent(self) -> Agent:
        return build_agent(self.db, FixedStatusPage(STATUS))

    def query(self, query: str) -> Any:
        with closing(sqlite3.connect(self.db)) as con:
            return con.execute(query).fetchone()[0]

    def snapshot(self) -> dict[str, Any]:
        with closing(sqlite3.connect(self.db)) as con:
            return {"tickets": con.execute("SELECT id, status, priority, team FROM tickets").fetchall(),
                    "replies": con.execute("SELECT ticket_id, body FROM replies").fetchall()}
