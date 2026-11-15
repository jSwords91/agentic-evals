"""The agent: one model, one loop, any tools.

    agent = Agent(system="You help with ...", tools=[convert_time],
                  mcp_servers=[stdio("python", "-m", "my_server")])
    async with agent.session() as session:
        reply = await session.send("What's on Thursday?")

`Agent` is a definition: the prompt, the model, the tools and the limits. It's immutable, so one
definition can be shared and run many times, and variants never touch the original:

    agent.replace(model="claude-sonnet-5-5", effort="low")
    agent.without_tools("send_email")       # an ablation: does this tool earn its place?

`Session` is one conversation with an agent. It owns the MCP connections, the message history,
the record of tool calls and the bill. Every eval trial gets a fresh one.

A3 [BEA][HD]: the simplest design that works. One model, one context window, tools in a loop.
  No router, planner or subagents. Every extra component encodes an assumption about what the
  model can't do. Add one when an eval shows the need, and keep the eval that justified it.
A4 [MA]: the brain (model and prompt), the hands (tools and MCP servers) and the session
  (history and logs) are separate, so any one can change without the others.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

from .mcp import connect
from .screen import REDACTED, UNTRUSTED, Screen, Verdict, check
from .skills import load_skill_tool, load_skills, skills_prompt
from .tools import Tool, ToolError
from .usage import Usage, check_priced

# A refusal is re-run on another model inside the same API call (server-side fallback). The
# "default" form routes by refusal category, so there's no list of fallback models to maintain.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class Agent:
    system: str
    tools: Sequence[Tool] = ()
    mcp_servers: Sequence[Any] = ()      # anything mcp.Client accepts; see mcp.py
    skills: str | Path | None = None     # a folder of */SKILL.md files; see skills.py
    screen: Screen | None = None         # the injection screen; see screen.py
    # K1 [docs]: model and effort are configuration, so trying another model is a flag on an
    #   eval run, not a code change. Opus 5.5 defaults to medium effort. It's set explicitly so
    #   that switching models doesn't silently change it.
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_tokens: int = 16000  # per response; a safe ceiling for non-streaming calls
    # A2 [HD][CONTAIN]: hard stops enforced by the harness, not requested of the model.
    max_turns: int = 20
    max_cost_usd: float = 1.00
    # E10 [docs]: Opus 5.5 always thinks, but by default returns the thinking empty. "summarized"
    #   puts a readable summary of its reasoning into every transcript, at no extra cost.
    thinking_display: str = "summarized"
    excluded_tools: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "mcp_servers", tuple(self.mcp_servers))
        object.__setattr__(self, "excluded_tools", frozenset(self.excluded_tools))
        check_priced(self.model)
        if self.screen:
            check_priced(self.screen.model)

    def replace(self, **changes: Any) -> Agent:
        return dataclasses.replace(self, **changes)

    def with_tools(self, *tools: Tool) -> Agent:
        return self.replace(tools=(*self.tools, *tools))

    def without_tools(self, *names: str) -> Agent:
        """Leave out tools by name, local or MCP. Checked when a session starts."""
        return self.replace(excluded_tools=self.excluded_tools | set(names))

    def session(self, client: anthropic.AsyncAnthropic | None = None) -> Session:
        return Session(self, client)


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool


@dataclass(frozen=True)
class Reply:
    text: str
    # "end_turn" when the model finished. Otherwise why it stopped early: "max_turns" or
    # "max_cost" (the harness's limits), or "refusal" or "max_tokens" (the API's).
    stop_reason: str

    @property
    def completed(self) -> bool:
        return self.stop_reason == "end_turn"


class Session:
    """One conversation with an agent. Use as `async with agent.session() as session:`."""

    def __init__(self, agent: Agent, client: anthropic.AsyncAnthropic | None = None):
        self.agent = agent
        self.client = client or anthropic.AsyncAnthropic()
        self.system = agent.system
        self.tools: dict[str, Tool] = {}
        self.messages: list[dict[str, Any]] = []  # C4: append-only
        self.tool_log: list[ToolCall] = []        # every call, in order: what graders read
        self.redactions: list[dict[str, str]] = []  # text the screen removed, and why
        self.usage = Usage(agent.model)
        self.screen_usage = Usage(agent.screen.model if agent.screen else agent.model)
        self._verdicts: dict[str, Verdict] = {}   # the screen checks each distinct span once
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> Session:
        async with AsyncExitStack() as stack:  # if setup fails, connections opened so far close
            tools = list(self.agent.tools)
            for server in self.agent.mcp_servers:
                tools += await connect(server, stack)
            if self.agent.skills is not None:
                skills = load_skills(self.agent.skills)
                tools.append(load_skill_tool(skills))
                self.system += "\n\n" + skills_prompt(skills)
            self.tools = _select(tools, self.agent.excluded_tools)
            self._stack = stack.pop_all()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    @property
    def cost_usd(self) -> float:
        return self.usage.cost_usd + self.screen_usage.cost_usd

    async def send(self, text: str) -> Reply:
        """Send one user message and loop until the model replies or a limit stops it.

        Call it again with the next message to continue the same conversation.
        """
        self.messages.append({"role": "user", "content": text})
        for _ in range(self.agent.max_turns):
            if self.cost_usd >= self.agent.max_cost_usd:
                return Reply("", "max_cost")
            response = await self.client.beta.messages.create(**self.request())
            self.usage.add(response.usage)
            # C4 [docs]: the reply goes into the history exactly as it came back, thinking
            #   blocks included. Editing or dropping earlier turns would break the prompt cache
            #   and invalidate the model's earlier thinking.
            self.messages.append({"role": "assistant", "content": response.content})
            text = "".join(b.text for b in response.content if b.type == "text")
            calls = [b for b in response.content if b.type == "tool_use"]

            if response.stop_reason == "end_turn":
                return Reply(text, "end_turn")
            if response.stop_reason != "tool_use":
                # A6 [docs]: a refusal, or a response cut off at max_tokens, ends the turn. Its
                #   tool calls are answered with errors, not run, so the history is still a
                #   valid conversation to continue or hand to a person.
                if calls:
                    self.messages.append({"role": "user", "content": [
                        _result(b.id, "Not run: the response ended early.", True) for b in calls]})
                return Reply(text, response.stop_reason)

            # A1 [FOLD][docs]: run the whole batch concurrently and return every result in ONE
            #   message. Splitting results across messages teaches the model not to batch.
            records = await asyncio.gather(*(self._run_tool(b.name, b.input) for b in calls))
            self.tool_log += records
            self.messages.append({"role": "user", "content": [
                _result(b.id, r.output, r.is_error) for b, r in zip(calls, records, strict=True)]})
        return Reply("", "max_turns")

    async def call_tool(self, name: str, args: dict[str, Any]) -> ToolCall:
        """Run one tool directly, without the model. Used to replay reference solutions (E9)."""
        record = await self._run_tool(name, args)
        self.tool_log.append(record)
        return record

    def request(self) -> dict[str, Any]:
        """The Messages API request for the next turn."""
        request: dict[str, Any] = {
            "model": self.agent.model,
            "max_tokens": self.agent.max_tokens,
            "system": self.system,
            "messages": list(self.messages),
            "thinking": {"type": "adaptive", "display": self.agent.thinking_display},
            "output_config": {"effort": self.agent.effort},
            # C3 [docs]: cache the conversation prefix. Tools and system prompt never change
            #   within a session, so every call after the first reads them from the cache.
            "cache_control": {"type": "ephemeral"},
        }
        if self.tools:
            request["tools"] = [t.to_api() for t in self.tools.values()]
        if self.agent.model in FALLBACK_MODELS:
            request |= {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        return request

    async def _run_tool(self, name: str, args: dict[str, Any]) -> ToolCall:
        tool = self.tools.get(name)
        try:
            if tool is None:
                raise ToolError(f"There is no tool called {name!r}.")
            output, is_error = await tool.run(args), False
        except ToolError as e:
            output, is_error = str(e), True
        except Exception as e:
            # A5 [TOOLS]: a broken tool becomes an error the model can see and work around,
            #   not a crashed conversation.
            output, is_error = f"{name} failed: {type(e).__name__}: {e}", True
        if self.agent.screen:
            output = await self._screen(output)
        return ToolCall(name, args, output, is_error)

    async def _screen(self, text: str) -> str:
        """S7: replace every marked span that the screen says is aimed at an AI."""
        spans = list(dict.fromkeys(UNTRUSTED.findall(text)))
        verdicts = await asyncio.gather(*(self._check(span) for span in spans))
        for span, verdict in zip(spans, verdicts, strict=True):
            if verdict.contains_instructions:
                text = text.replace(f"<untrusted>{span}</untrusted>",
                                    f"<untrusted>{REDACTED}</untrusted>")
        return text

    async def _check(self, span: str) -> Verdict:
        if span in self._verdicts:
            return self._verdicts[span]
        assert self.agent.screen is not None
        try:
            verdict, usage = await check(self.client, self.agent.screen, span)
        except Exception as e:
            # Fail closed: if the screen can't run, for any reason, the text doesn't get
            #   through. The broad except is deliberate, and only here: a guard that crashes
            #   the conversation, or waves text through on an unexpected error, isn't a guard.
            #   Not cached, so the next sight of the same text gets a fresh check.
            self.redactions.append({"text": span, "reason": f"screen unavailable: {type(e).__name__}"})
            return Verdict(reason="The screen couldn't run.", contains_instructions=True)
        self.screen_usage.add(usage)
        self._verdicts[span] = verdict
        if verdict.contains_instructions:
            self.redactions.append({"text": span, "reason": verdict.reason})
        return verdict


def _select(tools: list[Tool], excluded: frozenset[str]) -> dict[str, Tool]:
    names = [t.name for t in tools]
    if duplicates := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"More than one tool is called {', '.join(duplicates)}.")
    if unknown := sorted(excluded - set(names)):
        raise ValueError(f"Can't exclude {', '.join(unknown)}: no tool has that name.")
    # C3: a fixed order. The tool list is the first thing in every request, so any change to it
    #   (even its order) would invalidate the whole cache.
    return {t.name: t for t in sorted(tools, key=lambda t: t.name) if t.name not in excluded}


def _result(tool_use_id: str, content: str, is_error: bool) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content,
            "is_error": is_error}
