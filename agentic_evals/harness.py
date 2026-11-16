"""The harness: the loop that calls Claude and routes its tool calls to the hands.

    session = Session()
    async with Hands(tools) as hands:
        harness = Harness(agent, hands, session)
        reply = await harness.send("What's on Thursday?")

A4 [MA]: the brain is "Claude and its harness". `Agent` says what the brain is told and allowed
  to do; this is the part that does it. It keeps no state of its own: it reads the conversation
  from the session and writes every step back, so it can be thrown away at any point and a new
  one given the same session carries on (`resume`).
A3 [BEA][HD]: the simplest design that works. One model, one context window, tools in a loop.
  No router, planner or subagents. Every extra component encodes an assumption about what the
  model can't do. Add one when an eval shows the need, and keep the eval that justified it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import anthropic

from .agent import Agent
from .hands import Hands
from .screen import REDACTED, UNTRUSTED, Verdict, check
from .session import ModelResponse, ScreenCheck, Session, Stopped, Tokens, ToolResult, UserMessage, calls
from .skills import load_skill_tool, load_skills, skills_prompt
from .tools import Tool, execute, index

# A refusal is re-run on another model inside the same API call (server-side fallback). The
# "default" form routes by refusal category, so there's no list of fallback models to maintain.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

NOT_RUN = "Not run: the response ended early."
INTERRUPTED = ("The session was interrupted while this tool was running, so it may or may not have "
               "taken effect. Check before running it again.")


@dataclass(frozen=True)
class Reply:
    text: str
    # "end_turn" when the model finished. Otherwise why it stopped early: "max_turns" or
    # "max_cost" (the harness's limits), or "refusal" or "max_tokens" (the API's).
    stop_reason: str

    @property
    def completed(self) -> bool:
        return self.stop_reason == "end_turn"


class Harness:
    def __init__(self, agent: Agent, hands: Hands, session: Session,
                 client: anthropic.AsyncAnthropic | None = None):
        self.agent, self.hands, self.session = agent, hands, session
        self.client = client or anthropic.AsyncAnthropic()
        self.system = agent.system
        # The brain's own tools. Loading a skill is the agent reading its own instructions, not
        #   acting on the world, so it belongs here rather than in the hands (T6, mcp.py).
        own: list[Tool] = []
        if agent.skills is not None:
            skills = load_skills(agent.skills)
            own.append(load_skill_tool(skills))
            self.system += "\n\n" + skills_prompt(skills)
        self.own_tools = index(own)
        index([*hands.tools.values(), *own])  # a name used by both is a configuration error

    async def send(self, text: str) -> Reply:
        """Send one user message and loop until the model replies or a limit stops it.

        Call it again with the next message to continue the same conversation.
        """
        self.session.append(UserMessage(text=text))
        return await self._loop()

    async def resume(self) -> Reply:
        """Carry on from the session's last event: what a new harness does after a crash.

        Tool calls left without a result are answered as interrupted, not run again. A tool may
        have finished its work (sent the email, triaged the ticket) just before the crash, and
        running it twice could do the thing twice. The model is told the truth, that the outcome
        is unknown, and decides what to check. A session whose turn has ended is left as it is.
        """
        if not self.session.events:
            raise ValueError("The session is empty: there's nothing to resume.")
        for call in self.session.unanswered():
            self.session.append(ToolResult(tool_use_id=call["id"], name=call["name"], input=call["input"],
                                           output=INTERRUPTED, is_error=True, status="interrupted"))
        last = self.session.events[-1]
        if isinstance(last, Stopped):
            return Reply("", last.reason)
        if not isinstance(last, UserMessage):
            response = next(e for e in reversed(self.session.events) if isinstance(e, ModelResponse))
            if response.stop_reason != "tool_use":
                return Reply(_text(response), response.stop_reason)
        return await self._loop()

    def request(self) -> dict[str, Any]:
        """The Messages API request for the next turn, built from the session."""
        request: dict[str, Any] = {
            "model": self.agent.model,
            "max_tokens": self.agent.max_tokens,
            "system": self.system,
            "messages": self.session.messages(),
            "thinking": {"type": "adaptive", "display": self.agent.thinking_display},
            "output_config": {"effort": self.agent.effort},
            # C3 [docs]: cache the conversation prefix. Tools and system prompt never change
            #   within a session, so every call after the first reads them from the cache.
            "cache_control": {"type": "ephemeral"},
        }
        if tools := sorted([*self.hands.definitions(), *(t.to_api() for t in self.own_tools.values())],
                           key=lambda t: t["name"]):
            request["tools"] = tools
        if self.agent.model in FALLBACK_MODELS:
            request |= {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        return request

    async def _loop(self) -> Reply:
        # A2 [HD][CONTAIN]: hard stops enforced here, not requested of the model. Both are read
        #   from the session, so a resumed turn can't overrun them.
        turns = _turns_since_user_message(self.session.events)
        while turns < self.agent.max_turns:
            if self.session.cost_usd >= self.agent.max_cost_usd:
                self.session.append(Stopped(reason="max_cost"))
                return Reply("", "max_cost")
            api = await self.client.beta.messages.create(**self.request())
            # C4 [docs]: the response goes into the log as the SDK would send it back, thinking
            #   blocks included. Editing or dropping earlier turns would break the prompt cache
            #   and invalidate the model's earlier thinking.
            response = ModelResponse(model=self.agent.model, stop_reason=api.stop_reason,
                                     content=[b.to_dict(mode="json") for b in api.content],
                                     usage=Tokens.of(api.usage))
            self.session.append(response)
            turns += 1

            if response.stop_reason == "end_turn":
                return Reply(_text(response), "end_turn")
            if response.stop_reason != "tool_use":
                # A6 [docs]: a refusal, or a response cut off at max_tokens, ends the turn. Its
                #   tool calls are answered with errors, not run, so the history is still a
                #   valid conversation to continue or hand to a person.
                for call in calls(response):
                    self.session.append(ToolResult(tool_use_id=call["id"], name=call["name"], input=call["input"],
                                                   output=NOT_RUN, is_error=True, status="not_run"))
                return Reply(_text(response), response.stop_reason)

            # A1 [FOLD][docs]: run the whole batch concurrently. The session sends every result
            #   back in ONE message; splitting them across messages teaches the model not to batch.
            await asyncio.gather(*(self._run(call) for call in calls(response)))
        self.session.append(Stopped(reason="max_turns"))
        return Reply("", "max_turns")

    async def _run(self, call: dict[str, Any]) -> None:
        name, args = call["name"], call["input"]
        if name in self.own_tools:
            result = await execute(self.own_tools[name], name, args)
        else:
            result = await self.hands.execute(name, args)
        output = await self._screen(result.output) if self.agent.screen else result.output
        self.session.append(ToolResult(tool_use_id=call["id"], name=name, input=args,
                                       output=output, is_error=result.is_error))

    async def _screen(self, text: str) -> str:
        """S7: replace every marked span that the screen says is aimed at an AI."""
        spans = list(dict.fromkeys(UNTRUSTED.findall(text)))
        flagged = await asyncio.gather(*(self._check(span) for span in spans))
        for span, contains_instructions in zip(spans, flagged, strict=True):
            if contains_instructions:
                text = text.replace(f"<untrusted>{span}</untrusted>", f"<untrusted>{REDACTED}</untrusted>")
        return text

    async def _check(self, span: str) -> bool:
        """Whether a span tries to instruct an AI. Each distinct span is checked once per session."""
        screen = self.agent.screen
        assert screen is not None
        earlier = next((e for e in self.session.events
                        if isinstance(e, ScreenCheck) and e.text == span and e.usage is not None), None)
        if earlier:
            return earlier.contains_instructions
        try:
            verdict, usage = await check(self.client, screen, span)
        except Exception as e:
            # Fail closed: if the screen can't run, for any reason, the text doesn't get
            #   through. The broad except is deliberate, and only here: a guard that crashes
            #   the conversation, or waves text through on an unexpected error, isn't a guard.
            #   Logged without usage, so it isn't reused: the next sight of the text is checked again.
            verdict, tokens = Verdict(reason=f"screen unavailable: {type(e).__name__}",
                                      contains_instructions=True), None
        else:
            tokens = Tokens.of(usage)
        self.session.append(ScreenCheck(model=screen.model, text=span, reason=verdict.reason,
                                        contains_instructions=verdict.contains_instructions, usage=tokens))
        return verdict.contains_instructions


def _text(response: ModelResponse) -> str:
    return "".join(b["text"] for b in response.content if b["type"] == "text")


def _turns_since_user_message(events: tuple[Any, ...]) -> int:
    turns = 0
    for e in reversed(events):
        if isinstance(e, UserMessage):
            break
        turns += isinstance(e, ModelResponse)
    return turns
