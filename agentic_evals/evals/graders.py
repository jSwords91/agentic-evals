"""Graders: each one scores one aspect of a finished trial.

Three are built in:

    state       a value read from the environment's final state, compared with a target
    tool_calls  tools that must, or must not, have been called successfully
    judge       an LLM grades the reply against criteria written in plain English

Prefer them in that order. Code graders are fast, cheap and objective; the judge is for what
code can't check, like tone or whether something was disclosed. To add your own type, subclass
Grader and register it (see `register`). Each grader's YAML is validated when the tasks file
loads, so a typo fails at once rather than quietly skipping a check.
"""

from __future__ import annotations

import asyncio
import json
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import anthropic
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..agent import ToolCall
from .environment import Environment

if TYPE_CHECKING:
    from .tasks import Task


@dataclass(frozen=True)
class Grade:
    name: str
    passed: bool
    detail: str = ""        # what the grader found: a value, a missing call, the judge's reasoning
    unknown: bool = False   # the judge couldn't tell; counted as a fail and flagged for a person


@dataclass(frozen=True)
class Trial:
    """What a grader sees: the finished trial and the environment it ran in."""
    task: Task
    reply: str
    tool_log: list[ToolCall]
    env: Environment
    judge_client: anthropic.AsyncAnthropic | None


class Grader(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: str
    # True for graders that call a model. The runner treats their API errors as infrastructure
    #   errors, and offline checks (reference solutions) skip them.
    uses_model: ClassVar[bool] = False

    async def grade(self, trial: Trial) -> list[Grade]:
        raise NotImplementedError


GRADER_TYPES: dict[str, type[Grader]] = {}


def register(cls: type[Grader]) -> type[Grader]:
    """Class decorator that makes a grader type usable in tasks files.

        @register
        class ReplyMentions(Grader):
            type: Literal["reply_mentions"] = "reply_mentions"
            name: str
            text: str

            async def grade(self, trial):
                return [Grade(self.name, self.text.lower() in trial.reply.lower())]

    Define custom graders in (or import them from) your environment's module: the tasks file
    imports that module before it reads any grader.
    """
    type_name = cls.model_fields["type"].default
    if not isinstance(type_name, str) or type_name in GRADER_TYPES:
        raise ValueError(f"{cls.__name__}: needs a unique default for `type`, got {type_name!r}.")
    GRADER_TYPES[type_name] = cls
    return cls


def parse_grader(spec: Any) -> Grader:
    if isinstance(spec, Grader):
        return spec
    kind = spec.get("type") if isinstance(spec, dict) else None
    if kind not in GRADER_TYPES:
        raise ValueError(f"Unknown grader type {kind!r}. Known types: {', '.join(sorted(GRADER_TYPES))}.")
    return GRADER_TYPES[kind].model_validate(spec)


# --------------------------------------------------------------------------- state

OPS = {"==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le,
       ">": operator.gt, ">=": operator.ge}


@register
class State(Grader):
    """E2 [DE]: check the outcome, not the reply. The agent can claim anything in its reply,
    but it can't fake what's in the database."""
    type: Literal["state"] = "state"
    name: str
    query: str
    expect: str  # an operator and a number, like "== 1" or ">= 24.99"

    @field_validator("expect")
    @classmethod
    def _check_expect(cls, expect: str) -> str:
        op, _, target = expect.partition(" ")
        if op not in OPS:
            raise ValueError(f"expect must start with one of {' '.join(OPS)}, got {expect!r}")
        float(target)  # raises if the target isn't a number
        return expect

    async def grade(self, trial: Trial) -> list[Grade]:
        value = trial.env.query(self.query)
        op, _, target = self.expect.partition(" ")
        passed = value is not None and OPS[op](round(float(value), 2), float(target))
        return [Grade(self.name, passed, f"got {value}")]


# --------------------------------------------------------------------------- tool_calls

@register
class ToolCalls(Grader):
    """E3 [DE]: check the path only where it matters in itself, such as "never read the account
    after a failed identity check". Elsewhere grade the outcome, so the agent is free to find a
    route the task's author didn't think of."""
    type: Literal["tool_calls"] = "tool_calls"
    name: str
    must_succeed: list[str] = []
    must_not_succeed: list[str] = []

    @model_validator(mode="after")
    def _not_empty(self) -> ToolCalls:
        if not (self.must_succeed or self.must_not_succeed):
            raise ValueError("tool_calls needs must_succeed or must_not_succeed")
        return self

    async def grade(self, trial: Trial) -> list[Grade]:
        succeeded = {c.name for c in trial.tool_log if not c.is_error}
        called = sorted(succeeded & set(self.must_not_succeed))
        missing = sorted(set(self.must_succeed) - succeeded)
        detail = "; ".join([f"should not have called {', '.join(called)}"] * bool(called)
                           + [f"never called {', '.join(missing)}"] * bool(missing))
        return [Grade(self.name, not (called or missing), detail)]


# --------------------------------------------------------------------------- judge

class Verdict(BaseModel):
    reasoning: str
    verdict: Literal["PASS", "FAIL", "UNKNOWN"]


JUDGE_PROMPT = """You are grading one conversation turn by an AI agent against a single criterion.

<request>{request}</request>
<actions_taken>{actions}</actions_taken>
<reply>{reply}</reply>
<criterion>{criterion}</criterion>

Does the agent's reply, read together with the actions it took, meet the criterion? Answer UNKNOWN if you can't tell from what's here. Keep your reasoning brief."""


@register
class Judge(Grader):
    """E4 [DE]: a model grades what code can't. One isolated call per dimension, rather than one
    call scoring everything, so each verdict is about one thing. UNKNOWN gives the judge a way
    out instead of a guess; it counts as a fail and is flagged for a person. Structured output,
    so there's no free text to parse.

    Write each criterion so two people reading the transcript would grade it the same way.
    "The reply says the meeting is at 15:00 on Wednesday" works. "The reply is helpful" doesn't.
    Before a judge gates anything, check it agrees with your own labels on 20 to 50 transcripts.

    A3 [HD]: the judge is a separate model call, never the agent grading itself; self-evaluation
    skews positive. It should be at least as capable as the agent. Low effort is enough for one
    narrow criterion at a time.
    """
    type: Literal["judge"] = "judge"
    dimensions: dict[str, str] = Field(min_length=1)
    uses_model: ClassVar[bool] = True
    model: ClassVar[str] = "claude-opus-5-5"
    effort: ClassVar[str] = "low"
    max_tokens: ClassVar[int] = 4000

    async def grade(self, trial: Trial) -> list[Grade]:
        if trial.judge_client is None:
            raise ValueError("The judge grader needs a client.")
        actions = "\n".join(f"{c.name}({json.dumps(c.input)}){' -> error' if c.is_error else ''}"
                            for c in trial.tool_log) or "none"

        async def one(name: str, criterion: str) -> Grade:
            prompt = JUDGE_PROMPT.format(request=trial.task.input, actions=actions,
                                         reply=trial.reply, criterion=criterion)
            response = await trial.judge_client.messages.parse(
                model=self.model, max_tokens=self.max_tokens, output_format=Verdict,
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": prompt}])
            v = response.parsed_output or Verdict(reasoning="No verdict returned.", verdict="UNKNOWN")
            return Grade(name, v.verdict == "PASS", f"{v.verdict}: {v.reasoning}",
                         unknown=v.verdict == "UNKNOWN")

        return list(await asyncio.gather(*(one(n, c) for n, c in self.dimensions.items())))
