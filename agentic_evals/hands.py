"""The hands: the tools an agent acts with, behind one interface.

    async with Hands([convert_time, stdio("python", "-m", "my_server")]) as hands:
        hands.definitions()                         # what the model is told it can call
        await hands.execute("convert_time", {...})  # a name and input in, a result out

A4 [MA]: the hands are "sandboxes and tools that perform actions". Whatever sits behind them (a
  pure function, a database, an MCP server in another process) is reached the same way, so the
  harness never knows which, and any one can be swapped without touching the brain.
S1 [MA][CONTAIN]: anything a tool needs, such as a connection or a credential, is bound when the
  tool is built (see examples/helpdesk/tools.py). The harness and the session never see it.

Hands know nothing about the model, the prompt or the conversation. That's why an eval can run
a task's reference solution on them with no model at all (evals/runner.py, `replay`).
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AsyncExitStack
from typing import Any

from .mcp import connect
from .tools import Tool, ToolCall, execute, index


class Hands:
    """A set of tools: `Tool`s and MCP servers. Use as `async with Hands(tools) as hands:`.

    Entering connects to the MCP servers and lists their tools; leaving closes the connections.
    """

    def __init__(self, tools: Sequence[Any] = ()):
        self._sources = tuple(tools)
        self.tools: dict[str, Tool] = {}
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> Hands:
        async with AsyncExitStack() as stack:  # if setup fails, connections opened so far close
            tools: list[Tool] = []
            for source in self._sources:
                tools += [source] if isinstance(source, Tool) else await connect(source, stack)
            self.tools = index(tools)
            self._stack = stack.pop_all()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    def definitions(self) -> list[dict[str, Any]]:
        """The tool definitions for the Messages API, in a fixed order (C3)."""
        return [t.to_api() for t in self.tools.values()]

    async def execute(self, name: str, args: dict[str, Any]) -> ToolCall:
        """Run one tool. Never raises: a failure comes back as an error result (A5)."""
        return await execute(self.tools.get(name), name, args)
