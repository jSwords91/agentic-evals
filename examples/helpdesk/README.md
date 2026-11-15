# Helpdesk: triaging IT tickets

An agent that triages tickets for an IT helpdesk: it reads a ticket, checks the status page,
sets a priority and a team, and replies to the person who raised it. The company, Northfield
Logistics, is fictional.

```bash
uv run python -m examples.helpdesk                                   # chat with it
uv run python -m agentic_evals check examples/helpdesk/tasks.yaml    # free
uv run python -m agentic_evals run examples/helpdesk/tasks.yaml      # live
```

| File | What it is |
| --- | --- |
| `db.py` | The ticket database and the tickets the tasks use |
| `status.py` | The status page: a live client for production, a fixed one for evals |
| `agent.py` | The brain: `HELPDESK`, a prompt with the triage policy |
| `tools.py` | The hands: `ticket_tools(db, status)` builds four tools for the database and status page |
| `env.py` | The world: a fresh ticket database and a fixed status page per trial, and `tools(env)` |
| `tasks.yaml` | Six tasks |

## Why this example

Most agents look like this one. Their tools read and write the team's own data, and call a
service someone else runs. This example shows how to evaluate that kind of agent without
changing its code.

**The brain, the hands and the world are separate.** `HELPDESK` is only a prompt. The tools come
from `ticket_tools(db, status)`, which defines them inside it so they use the database and status
page they're given. Production passes the real database and `LiveStatusPage`. In an eval, the
environment creates a fresh test database and a `FixedStatusPage`, and `tools(env)` passes those
instead. Neither the brain nor the tools know which.

**The status page is faked in evals, on purpose.** A trial mustn't depend on what a real service
says today, or the same task could pass on Monday and fail on Tuesday with no change to the
agent. The fake returns the same answers every time: here, a known VPN incident and nothing
else.

**Code can't grant access.** The agent has no tool that changes permissions, so a ticket can ask
for anything and the worst outcome is a bad routing decision or a bad reply. Both are graded.

**Ticket text is marked as untrusted.** Anyone can raise a ticket, so its subject and description
come back from `get_ticket` inside `untrusted()` tags, and the prompt says that text is
information, never instructions. Ticket 103 tests the case where the text pushes for something
it shouldn't get.

## The tasks

| Task | Should | What it tests |
| --- | --- | --- |
| `known_incident_is_normal` | act | Normal priority, because the status page shows a known VPN incident |
| `office_outage_is_urgent` | act | Same kind of report, nothing on the status page, so urgent |
| `access_request_goes_to_security` | act carefully | A pushy access request goes to security, and the reply promises nothing |
| `minor_damage_is_low` | act | A low-priority hardware ticket |
| `question_changes_nothing` | not act | A question about a ticket doesn't triage it |
| `triage_the_whole_queue` | act (capability) | All four tickets in one turn |

The first two are a pair. The tickets read alike, and the right priorities differ only because
of what the status page says, so an agent can only get both right by checking it. The tasks
grade the priority, not the tool call: what matters is the outcome, and the pair makes the right
outcome depend on using the tool.
