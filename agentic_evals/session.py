"""The session: an append-only log of everything that happened in one conversation.

    session = Session()                        # in memory, for one process
    session = Session("runs/maya.jsonl")       # on disk: survives the process, and resumes

A4 [MA]: the session is "the append-only log of everything that happened". It lives outside the
  harness, so the harness keeps nothing that must survive a crash: a new harness given the same
  session carries on from the last event (harness.py, `resume`). Everything else is read from the
  log: the messages sent to the model, the tool calls graders check, the bill.
C4 [docs]: each model response is stored exactly as the API sends it back in a request, thinking
  blocks included. So the messages rebuilt from the log are byte-for-byte what the first harness
  sent, and a resumed conversation still hits the prompt cache.

An event is written to disk before it's added in memory, so the log never claims something the
file doesn't hold. There's one writer per session; two harnesses on one log would interleave.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .tools import ToolCall
from .usage import Usage


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Tokens(_Record):
    """A response's token counts, under the API's own field names."""
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def of(cls, usage: Any) -> Tokens:
        return cls(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                   cache_read_input_tokens=usage.cache_read_input_tokens or 0,
                   cache_creation_input_tokens=usage.cache_creation_input_tokens or 0)


class UserMessage(_Record):
    type: Literal["user_message"] = "user_message"
    text: str


class ModelResponse(_Record):
    type: Literal["model_response"] = "model_response"
    model: str
    content: list[dict[str, Any]]   # the content blocks, as the SDK would send them back
    stop_reason: str
    usage: Tokens


class ToolResult(_Record):
    """One tool call's result, logged the moment that tool finishes.

    One event per call, not per batch: if the process dies mid-batch, the calls that finished
    keep their results, and only those still running are in doubt. `status` says who wrote it:
    "ran" for the tool itself, otherwise the harness, answering for a call it couldn't run
    ("not_run") or one whose outcome a crash left unknown ("interrupted").
    """
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool
    status: Literal["ran", "not_run", "interrupted"] = "ran"


class ScreenCheck(_Record):
    """One span of third-party text, checked by the injection screen (screen.py)."""
    type: Literal["screen_check"] = "screen_check"
    model: str
    text: str
    contains_instructions: bool
    reason: str
    usage: Tokens | None   # None when the screen couldn't run, and the text was removed anyway


class Stopped(_Record):
    """The harness ended the turn at one of its own limits (A2)."""
    type: Literal["stopped"] = "stopped"
    reason: Literal["max_turns", "max_cost"]


Event = Annotated[UserMessage | ModelResponse | ToolResult | ScreenCheck | Stopped,
                  Field(discriminator="type")]
_EVENT: TypeAdapter[Event] = TypeAdapter(Event)


class Session:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self._events: list[Event] = []
        if self.path is not None and self.path.exists():
            text = self.path.read_text()
            if not text.endswith("\n"):
                # A crash mid-write leaves a torn last line. That event was never added in
                #   memory, so it never happened: drop it, and the log ends at its last whole event.
                text = text[:text.rfind("\n") + 1]
                self.path.write_text(text)
            self._events = [_EVENT.validate_json(line) for line in text.splitlines()]

    def append(self, event: Event) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(event.model_dump_json() + "\n")
        self._events.append(event)

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    # ------------------------------------------------------------ views, all read from the log

    def messages(self) -> list[dict[str, Any]]:
        """The conversation in the Messages API's format."""
        messages: list[dict[str, Any]] = []
        for event, results in self._turns():
            if isinstance(event, UserMessage):
                messages.append({"role": "user", "content": event.text})
                continue
            messages.append({"role": "assistant", "content": event.content})
            if results:  # A1: every result of a batch in one message, in the order of the calls
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r.tool_use_id, "content": r.output,
                     "is_error": r.is_error} for r in results]})
        return messages

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Every tool that ran, in the order the model asked for them. What graders read."""
        return [ToolCall(r.name, r.input, r.output, r.is_error)
                for _, results in self._turns() for r in results if r.status == "ran"]

    def unanswered(self) -> list[dict[str, Any]]:
        """Tool calls in the last response that have no result yet: the work a crash cut off."""
        last = next((e for e in reversed(self._events) if isinstance(e, ModelResponse)), None)
        if last is None or last.stop_reason != "tool_use":
            return []
        answered = {e.tool_use_id for e in self._events if isinstance(e, ToolResult)}
        return [b for b in calls(last) if b["id"] not in answered]

    def _turns(self) -> Iterator[tuple[UserMessage | ModelResponse, list[ToolResult]]]:
        """Each user message, and each model response with its tool results in call order.
        Results are logged as they finish, so the log holds them in finishing order."""
        results = {e.tool_use_id: e for e in self._events if isinstance(e, ToolResult)}
        for e in self._events:
            if isinstance(e, UserMessage):
                yield e, []
            elif isinstance(e, ModelResponse):
                yield e, [results[b["id"]] for b in calls(e) if b["id"] in results]

    @property
    def redactions(self) -> list[dict[str, str]]:
        """Text the screen removed, and why."""
        return [{"text": e.text, "reason": e.reason}
                for e in self._events if isinstance(e, ScreenCheck) and e.contains_instructions]

    @property
    def usage(self) -> Usage:
        """The main model's usage."""
        return self._usage(ModelResponse)

    @property
    def screen_usage(self) -> Usage:
        return self._usage(ScreenCheck)

    @property
    def cost_usd(self) -> float:
        return self.usage.cost_usd + self.screen_usage.cost_usd

    def _usage(self, kind: type[ModelResponse] | type[ScreenCheck]) -> Usage:
        usage = Usage()
        for e in self._events:
            if isinstance(e, kind) and e.usage is not None:
                usage.add(e.usage, e.model)
        return usage


def calls(response: ModelResponse) -> list[dict[str, Any]]:
    """The tool calls a response asked for."""
    return [b for b in response.content if b["type"] == "tool_use"]
