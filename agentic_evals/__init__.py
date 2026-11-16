"""A Claude agent and the task-based evals that test it, written to be read and adapted. Read it in this order:

    tools.py    @tool: typed Python functions the model can call
    mcp.py      tools from MCP servers
    hands.py    Hands: the tools, behind one interface
    agent.py    Agent: the brain's configuration
    skills.py   procedures loaded on demand
    screen.py   the injection screen
    session.py  Session: the append-only log of a conversation
    harness.py  Harness: the loop, joining a brain to its hands and writing to the session
    evals/      tasks, graders, metrics and the runner
"""

from .agent import Agent
from .chat import chat
from .hands import Hands
from .harness import Harness, Reply
from .mcp import stdio
from .screen import Screen, untrusted
from .session import Session
from .tools import Tool, ToolCall, ToolError, tool

__all__ = ["Agent", "Hands", "Harness", "Reply", "Screen", "Session", "Tool", "ToolCall", "ToolError",
           "chat", "stdio", "tool", "untrusted"]
