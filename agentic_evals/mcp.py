"""MCP: tools that live in another process or on another machine.

An agent takes any server that `mcp.Client` can connect to:

    stdio("python", "-m", "my_server")    a local command, spoken to over stdin/stdout
    "https://example.com/mcp"             a remote server, over Streamable HTTP
    my_server                             an MCPServer object, in-process (useful in tests)

T6 [docs][CEM]: MCP is the standard way to expose a system to agents. Implement a server once
  and any MCP client can use it: this agent, Claude Desktop, Claude Code, someone else's agent.
  Use it for systems your agent doesn't own (an inbox, a calendar, a database). Keep the agent's
  own concerns, like loading a skill, as local tools: a process boundary there buys nothing.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from .tools import Tool, ToolError


def stdio(command: str, *args: str, env: dict[str, str] | None = None,
          cwd: str | Path | None = None) -> StdioServerParameters:
    """A server started as a subprocess.

    S1 [CONTAIN]: the subprocess gets only `env` plus a few basics like PATH, not this process's
      environment, so your Anthropic API key never reaches it. Pass the server what it needs to
      do its job (a database path, its own credentials) and nothing more.
    """
    return StdioServerParameters(command=command, args=list(args), env=env,
                                 cwd=str(cwd) if cwd else None)


async def connect(server: Any, stack: AsyncExitStack) -> list[Tool]:
    """Connect to one MCP server and return its tools. The connection closes with `stack`."""
    client = await stack.enter_async_context(Client(server))
    return [_wrap(client, t) for t in (await client.list_tools()).tools]


def _wrap(client: Client, listed: Any) -> Tool:
    async def run(args: dict[str, Any]) -> str:
        result = await client.call_tool(listed.name, args)
        # Text only. Images and other content types would need converting to API content blocks.
        text = "\n".join(c.text for c in result.content if c.type == "text")
        if result.is_error:
            raise ToolError(text)
        return text

    return Tool(name=listed.name, description=listed.description or "",
                input_schema=listed.input_schema, run=run, strict=False)
