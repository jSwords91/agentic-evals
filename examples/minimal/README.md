# Minimal: the smallest agent and the smallest eval

A writing assistant with one tool, `word_count`.

```bash
uv run python -m examples.minimal                                    # chat with it
uv run python -m agentic_evals check examples/minimal/tasks.yaml     # free
uv run python -m agentic_evals run examples/minimal/tasks.yaml       # live
```

| File | What it is |
| --- | --- |
| `agent.py` | The brain: `WRITER`, a one-line prompt |
| `tools.py` | The hands: `word_count`, a pure function |
| `env.py` | The world: nothing in it. `tools(env)`, and a `reply_word_limit` grader |
| `tasks.yaml` | Three tasks |

The tool is a pure function, so there's nothing to set up before a trial and nothing to read
afterwards. Every grader reads the reply.

The tasks show how to choose a grader. Each fact is checked by the cheapest grader that can
check it reliably:

| Fact | Grader | Why |
| --- | --- | --- |
| The reply says "36" | `reply` | A number has one right form |
| The rewrite is under 10 words | `reply_word_limit` (custom) | Counting is exact; no built-in grader does it |
| The reply says the text is too long, or that it fits | `judge` | Meaning, worded many ways ("one over", "isn't short enough") |
| The rewrite keeps the meaning | `judge` | Meaning |

Counting words or matching a number with a judge would be slower, cost money, and sometimes be
wrong.

No task checks whether `word_count` was called. That's the route, not the outcome: if the model
counts correctly without it, the user is just as well served. If the tool helps, it shows up as
a higher pass rate on the counting tasks.

`too_long_for_a_listing` and `fits_in_a_bio` are a pair: the right answer is "no" for one and
"yes" for the other, so an agent that always gives the same answer fails one of them.
