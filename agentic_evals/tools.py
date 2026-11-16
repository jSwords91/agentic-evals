"""Tools: Python functions the model can call.

    @tool
    def convert_time(time: str, from_zone: str, to_zone: str) -> str:
        '''Convert a date and time between IANA time zones.'''
        ...

The decorator turns a typed function into a `Tool`: a name, a description, a JSON schema and
a way to run it. The model sees only the first three, so the docstring is the tool's
documentation for the model. Write it for a capable new colleague: what the tool does, when
to use it, and anything surprising about its arguments or results [TOOLS].

Tools from MCP servers (see mcp.py) become the same `Tool` type, so the agent loop never needs
to know where a tool lives. `execute` is the one way any tool is run.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import typing
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import anthropic
import pydantic


class ToolError(Exception):
    """An error the model should read and act on, such as "call verify_customer first".

    Raise it for expected refusals. The loop returns the message to the model unchanged.
    Any other exception is a bug in the tool; the model still sees it (as an error, so it can
    work around it) but prefixed with the exception type.
    """


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[[dict[str, Any]], Awaitable[str]]
    # T2 [docs]: strict mode guarantees the model's arguments match the schema. It's on for
    #   tools defined here, whose schemas we build. MCP servers own their schemas, which may
    #   not meet strict mode's rules, so their tools are sent as they are.
    strict: bool = True

    def to_api(self) -> dict[str, Any]:
        """The tool definition the Messages API expects."""
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema, "strict": self.strict}


@dataclass(frozen=True)
class ToolCall:
    """One tool run: what was asked for and what came back."""
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool


async def execute(tool: Tool | None, name: str, args: dict[str, Any]) -> ToolCall:
    """Run one tool call. Every outcome, including a crash, becomes output the model can read.

    A5 [TOOLS]: a broken tool becomes an error the model can see and work around, not a crashed
      conversation.
    """
    try:
        if tool is None:
            raise ToolError(f"There is no tool called {name!r}.")
        return ToolCall(name, args, await tool.run(args), False)
    except ToolError as e:
        return ToolCall(name, args, str(e), True)
    except Exception as e:
        return ToolCall(name, args, f"{name} failed: {type(e).__name__}: {e}", True)


def index(tools: Sequence[Tool]) -> dict[str, Tool]:
    """Tools by name, in a fixed order. Two tools with one name are a configuration error.

    C3: the order is fixed because the tool list is the first thing in every request, so any
      change to it (even its order) would invalidate the whole prompt cache.
    """
    names = [t.name for t in tools]
    if duplicates := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"More than one tool is called {', '.join(duplicates)}.")
    return {t.name: t for t in sorted(tools, key=lambda t: t.name)}


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    """Make a typed function into a Tool. Use as `@tool` or `@tool(name="...")`.

    Parameters become the input schema. Describe one with `Annotated[str, Field(description=...)]`
    when its name and type don't say enough. Sync functions run in a worker thread so they can't
    block other tool calls. A result that isn't a string is sent to the model as JSON.
    """

    def wrap(fn: Callable[..., Any]) -> Tool:
        description = inspect.getdoc(fn)
        if not description:
            raise ValueError(f"{fn.__name__} needs a docstring: it's the description the model reads.")
        hints = typing.get_type_hints(fn, include_extras=True)
        fields: dict[str, Any] = {}
        for p in inspect.signature(fn).parameters.values():
            if p.name not in hints:
                raise ValueError(f"{fn.__name__}: parameter {p.name!r} needs a type hint.")
            fields[p.name] = (hints[p.name], ... if p.default is p.empty else p.default)
        args_model = pydantic.create_model(f"{fn.__name__}_input",
                                           __config__=pydantic.ConfigDict(extra="forbid"), **fields)

        async def run(raw: dict[str, Any]) -> str:
            # Validated again here, even with strict mode on: strict mode can't express every
            # constraint (a minimum, a string length), and the SDK moves those out of the schema
            # sent to the API. The pydantic model still checks them.
            try:
                args = args_model.model_validate(raw)
            except pydantic.ValidationError as e:
                raise ToolError(f"Invalid arguments for {fn.__name__}: {e}") from None
            kwargs = {k: getattr(args, k) for k in fields}
            if inspect.iscoroutinefunction(fn):
                result = await fn(**kwargs)
            else:
                result = await asyncio.to_thread(fn, **kwargs)
            return result if isinstance(result, str) else json.dumps(result, default=str)

        # transform_schema is the SDK's own conversion to the subset of JSON Schema that strict
        # mode accepts. Reusing it beats keeping a copy of those rules here.
        return Tool(name=name or fn.__name__, description=description,
                    input_schema=anthropic.transform_schema(args_model), run=run)

    return wrap(fn) if fn is not None else wrap
