"""The status page: a service the helpdesk agent reads but doesn't own.

The agent's tool calls a `StatusPage`. In production that's `LiveStatusPage`, which makes an
HTTP request. In evals it's `FixedStatusPage`, which returns the same answer every time. A trial
must never depend on what a real service happens to say today, or the same task could pass on
Monday and fail on Tuesday with no change to the agent.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol


class StatusPage(Protocol):
    def status(self, service: str) -> str: ...


class LiveStatusPage:
    """What production would pass in. The URL and response shape are made up for the example."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def status(self, service: str) -> str:
        with urllib.request.urlopen(f"{self.base_url}/api/status/{service}", timeout=10) as response:
            return json.load(response)["status"]


class FixedStatusPage:
    """What evals pass in: canned answers, so every trial sees the same world."""

    def __init__(self, statuses: dict[str, str]):
        self.statuses = statuses

    def status(self, service: str) -> str:
        return self.statuses[service]
