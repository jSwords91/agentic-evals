"""Maya's assistant: the hands.

Two kinds of tool, each where it belongs:

  - the workspace (email, calendar, contacts): an MCP server, because it's a system the agent
    uses but doesn't own (T6). `workspace_tools(db)` starts it pointed at a workspace database.
  - convert_time: a local @tool, because it's a pure function that belongs to the agent.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from pydantic import Field

from agentic_evals import stdio, tool

REPO = Path(__file__).parent.parent.parent


@tool
def convert_time(
    time: Annotated[str, Field(description="Date and time, YYYY-MM-DD HH:MM")],
    from_zone: Annotated[str, Field(description="IANA zone, such as Europe/London")],
    to_zone: Annotated[str, Field(description="IANA zone, such as America/New_York")],
) -> str:
    """Convert a date and time from one time zone to another. Use it whenever a time is
    mentioned for someone in a different time zone; don't work it out yourself."""
    moment = datetime.fromisoformat(time).replace(tzinfo=ZoneInfo(from_zone))
    return moment.astimezone(ZoneInfo(to_zone)).strftime(f"%Y-%m-%d %H:%M ({to_zone})")


def workspace_tools(db: str | Path) -> list[Any]:
    """The workspace MCP server for the database at `db`, and convert_time."""
    workspace = stdio(sys.executable, "-m", "examples.assistant.server",
                      env={"WORKSPACE_DB": str(db)}, cwd=REPO)  # S1: only what it needs
    return [workspace, convert_time]
