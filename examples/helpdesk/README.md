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
| `agent.py` | The prompt (with the triage policy) and four tools, built by `build_agent(db, status)` |
| `env.py` | A fresh ticket database and a fixed status page for every trial |
| `tasks.yaml` | Six tasks |

## Why this example

Most agents look like this one. Their tools read and write the team's own data, and call a
service someone else runs. This example shows how to evaluate that kind of agent without
changing its code.

**The agent takes what its tools need as arguments.** `build_agent(db, status)` defines the
tools inside it, so they use the database and status page they're given. Production passes the
real database and `LiveStatusPage`. The eval environment passes a fresh test database and
`FixedStatusPage`.

**The status page is faked in evals, on purpose.** A trial mustn't depend on what a real service
says today, or the same task could pass on Monday and fail on Tuesday with no change to the
agent. The fake returns the same answers every time: here, a known VPN incident and nothing
else.

**Code can't grant access.** The agent has no tool that changes permissions, so a ticket can ask
for anything and the worst outcome is a bad routing decision or a bad reply. Both are graded.

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
