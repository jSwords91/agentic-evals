"""Shared test helpers. Every test is offline: a scripted model stands in for the API."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types.beta import BetaTextBlock, BetaToolUseBlock

from agentic_evals.evals import graders
from agentic_evals.screen import Verdict as ScreenVerdict
from examples.assistant.data import create_db

USAGE = SimpleNamespace(input_tokens=100, output_tokens=10, cache_read_input_tokens=50,
                        cache_creation_input_tokens=0)
API_DOWN = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.example"))


def response(stop_reason: str, *content: Any) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), usage=USAGE)


# Real SDK content blocks, so the harness serialises exactly what it would from the API.
def text(t: str) -> BetaTextBlock:
    return BetaTextBlock(type="text", text=t)


def tool_use(id: str, name: str, /, **input: Any) -> BetaToolUseBlock:
    return BetaToolUseBlock(type="tool_use", id=id, name=name, input=input)


def default_parse(request: dict) -> Any:
    """The screen never flags anything; the judge passes everything."""
    if request["output_format"] is ScreenVerdict:
        return ScreenVerdict(reason="describes an ordinary email", contains_instructions=False)
    return graders.Verdict(reasoning="looks right", verdict="PASS")


class ScriptedModel:
    """Stands in for AsyncAnthropic. Plays back canned responses and records every request.

    `responses` feed the agent loop, in order; the last one repeats. `parse` answers the
    structured-output calls (the screen and the judge): a function from request to result, or
    to an exception to raise.
    """

    def __init__(self, *responses: Any, parse=default_parse):
        self.responses, self.requests, self.parsed = list(responses), [], []
        self.parse_fn = parse
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.messages = SimpleNamespace(parse=self._parse)

    async def _create(self, **request: Any) -> Any:
        self.requests.append(request)
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r

    async def _parse(self, **request: Any) -> Any:
        self.parsed.append(request)
        out = self.parse_fn(request)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(parsed_output=out, usage=USAGE)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A fresh workspace database, with the server's tools pointed at it."""
    path = create_db(tmp_path / "workspace.db")
    monkeypatch.setenv("WORKSPACE_DB", str(path))
    return path
