"""The session is the log, and the harness keeps nothing else: a new harness resumes from it (A4)."""

from __future__ import annotations

import asyncio

import pytest

from agentic_evals import Agent, Hands, Harness, Session, tool
from agentic_evals.harness import INTERRUPTED
from agentic_evals.session import ModelResponse, ToolResult, UserMessage
from tests.conftest import API_DOWN, ScriptedModel, response, text, tool_use

AGENT = Agent(system="You are a test agent.")
sent: list[str] = []


@tool
def send_email(to: str) -> str:
    """Send an email. Not safe to repeat."""
    sent.append(to)
    return f"Sent to {to}."


async def harness_on(session: Session, model: ScriptedModel, then):
    async with Hands([send_email]) as hands:
        return await then(Harness(AGENT, hands, session, model))


def test_a_new_harness_resumes_a_crashed_turn_with_the_same_request(tmp_path):
    log = tmp_path / "session.jsonl"
    # The first harness runs a tool, then the API goes down: the process gives up mid-turn.
    first = ScriptedModel(response("tool_use", text("Sending."), tool_use("a", "send_email", to="sam")), API_DOWN)
    with pytest.raises(type(API_DOWN)):
        asyncio.run(harness_on(Session(log), first, lambda h: h.send("Email Sam.")))
    # A new process: a new session object read from disk, a new harness, a new client.
    second = ScriptedModel(response("end_turn", text("Done.")))
    reply = asyncio.run(harness_on(Session(log), second, lambda h: h.resume()))
    assert reply.text == "Done." and reply.completed
    # The request is exactly the one the first harness would have sent, so the cache still hits.
    assert second.requests[0] == first.requests[1]
    assert [e.type for e in Session(log)] == ["user_message", "model_response", "tool_result", "model_response"]


def test_calls_a_crash_cut_off_are_reported_as_unknown_not_run_again(tmp_path):
    sent.clear()
    session = Session(tmp_path / "session.jsonl")
    # The log as a crash mid-batch leaves it: two calls asked for, one result written.
    session.append(UserMessage(text="Email Sam and Alex."))
    session.append(ModelResponse(model=AGENT.model, stop_reason="tool_use",
                                 usage={"input_tokens": 1, "output_tokens": 1},
                                 content=[tool_use("a", "send_email", to="sam").to_dict(mode="json"),
                                          tool_use("b", "send_email", to="alex").to_dict(mode="json")]))
    session.append(ToolResult(tool_use_id="a", name="send_email", input={"to": "sam"}, output="Sent to sam.",
                              is_error=False))
    model = ScriptedModel(response("end_turn", text("Sam's sent; I'll check Alex's.")))
    asyncio.run(harness_on(session, model, lambda h: h.resume()))
    assert sent == []  # neither email is sent again
    results = model.requests[0]["messages"][2]["content"]
    assert [(r["tool_use_id"], r["is_error"]) for r in results] == [("a", False), ("b", True)]
    assert results[1]["content"] == INTERRUPTED
    assert [c.input["to"] for c in session.tool_calls] == ["sam"]  # graders see only what ran


def test_a_finished_turn_resumes_to_its_reply_without_calling_the_model():
    session = Session()
    asyncio.run(harness_on(session, ScriptedModel(response("end_turn", text("Hi."))), lambda h: h.send("hi")))
    model = ScriptedModel(response("end_turn", text("Again?")))
    reply = asyncio.run(harness_on(session, model, lambda h: h.resume()))
    assert reply.text == "Hi." and model.requests == []


def test_the_turn_budget_holds_across_a_resume(tmp_path):
    log = tmp_path / "session.jsonl"
    loop = response("tool_use", tool_use("a", "send_email", to="sam"))
    agent_of_two = Agent(system="x", max_turns=2)

    async def go(session, model, then):
        async with Hands([send_email]) as hands:
            return await then(Harness(agent_of_two, hands, session, model))

    with pytest.raises(type(API_DOWN)):
        asyncio.run(go(Session(log), ScriptedModel(loop, API_DOWN), lambda h: h.send("loop")))
    model = ScriptedModel(loop)
    reply = asyncio.run(go(Session(log), model, lambda h: h.resume()))
    assert reply.stop_reason == "max_turns" and len(model.requests) == 1  # one turn was already used


def test_a_write_torn_by_a_crash_is_dropped_and_the_log_carries_on(tmp_path):
    log = tmp_path / "session.jsonl"
    Session(log).append(UserMessage(text="Email Sam."))
    with log.open("a") as f:
        f.write('{"type": "model_resp')  # the process died here
    session = Session(log)
    assert [e.type for e in session] == ["user_message"]
    session.append(UserMessage(text="Are you there?"))
    assert [e.type for e in Session(log)] == ["user_message", "user_message"]
