"""A Claude agent and the task-based evals that test it, written to be read and adapted. Read it in this order:

    tools.py    @tool: typed Python functions the model can call
    mcp.py      tools from MCP servers
    skills.py   procedures loaded on demand
    screen.py   the injection screen
    agent.py    Agent and Session: the loop
    evals/      tasks, graders, metrics and the runner
"""

from .agent import Agent, Reply, Session, ToolCall
from .chat import chat
from .mcp import stdio
from .screen import Screen, untrusted
from .tools import Tool, ToolError, tool

__all__ = ["Agent", "Reply", "Screen", "Session", "Tool", "ToolCall", "ToolError", "chat",
           "stdio", "tool", "untrusted"]
