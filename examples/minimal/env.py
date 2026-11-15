"""The eval environment for the minimal agent, and one custom grader.

The agent's only tool is a pure function: it reads its input and changes nothing. So there's no
state to create before a trial and nothing to read afterwards. Its tasks are graded on which
tools it used (`tool_calls`) and on what the reply says (`reply`, `reply_word_limit`), and
`query` is never needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from agentic_evals import Agent
from agentic_evals.evals import Environment, Grade, Grader, Trial, register

from .agent import AGENT


class Stateless(Environment):
    def setup(self, workdir: Path) -> None:
        pass  # nothing to create

    def agent(self) -> Agent:
        return AGENT


@register
class ReplyWordLimit(Grader):
    """E14: a check the built-in graders can't express, written as code. Passes if the whole
    reply is at most `max_words` words. Counting words is a job for code, never for a judge."""
    type: Literal["reply_word_limit"] = "reply_word_limit"
    name: str
    max_words: int
    reads_reply: ClassVar[bool] = True

    async def grade(self, trial: Trial) -> list[Grade]:
        words = len(trial.reply.split())
        return [Grade(self.name, words <= self.max_words, f"{words} words")]
