"""A minimal terminal chat with any agent. Nothing is graded or saved; it's for trying things."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import anthropic

from .agent import Agent
from .hands import Hands
from .harness import Harness
from .session import Session


async def chat(agent: Agent, tools: Sequence[Any] = (), greeting: str = "") -> None:
    session = Session()
    async with Hands(tools) as hands:
        harness = Harness(agent, hands, session)
        print(greeting or "Chatting with the agent. Ctrl-D to quit.", end="\n\n")
        while True:
            try:
                text = await asyncio.to_thread(input, "you> ")
            except EOFError:
                break
            seen = len(session.tool_calls)
            try:
                reply = await harness.send(text)
            except anthropic.APIError as e:  # the SDK has already retried what it can
                print(f"\n[API error: {e}]\n")
                continue
            for call in session.tool_calls[seen:]:
                print(f"  - {call.name}" + (" (error)" if call.is_error else ""))
            print(f"\nagent> {reply.text}")
            if not reply.completed:
                print(f"  [stopped early: {reply.stop_reason}]")
            print(f"  [${session.cost_usd:.4f} so far, cache hit {session.usage.cache_hit_rate:.0%}]\n")
