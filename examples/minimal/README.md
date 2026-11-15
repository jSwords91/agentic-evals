# Minimal: the smallest agent and the smallest eval

A writing assistant with one tool, `word_count`.

```bash
uv run python -m examples.minimal                                    # chat with it
uv run python -m agentic_evals check examples/minimal/tasks.yaml     # free
uv run python -m agentic_evals run examples/minimal/tasks.yaml       # live
```

| File | What it is |
| --- | --- |
| `agent.py` | The tool and the agent: about ten lines |
| `env.py` | An environment with nothing in it, and a `reply_word_limit` grader |
| `tasks.yaml` | Three tasks |

The tool is a pure function, so there's nothing to set up before a trial and nothing to read
afterwards. Every grader looks at the tools the agent used or at its reply.

The tasks show how to choose a grader. Each fact is checked by the cheapest grader that can
check it reliably:

| Fact | Grader | Why |
| --- | --- | --- |
| The agent counted with the tool | `tool_calls` | The tool exists because models miscount, so the path matters |
| The reply says "36" | `reply` | A number has one right form |
| The rewrite is under 10 words | `reply_word_limit` (custom) | Counting is exact; no built-in grader does it |
| The reply says the text is too long | `judge` | Meaning, worded many ways ("one over", "isn't short enough") |
| The rewrite keeps the meaning | `judge` | Meaning |

Counting words or matching a number with a judge would be slower, cost money, and sometimes be
wrong.
