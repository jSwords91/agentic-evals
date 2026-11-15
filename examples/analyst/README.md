# Analyst: answering questions from a sales database

An agent that answers the commercial team's questions ("What was revenue in October?", "Which
products made the most profit?") by writing SQL against a sales warehouse. The company,
Brightleaf, is a fictional online plant shop.

```bash
uv run python -m examples.analyst                                    # ask it questions
uv run python -m agentic_evals check examples/analyst/tasks.yaml     # free
uv run python -m agentic_evals run examples/analyst/tasks.yaml       # live
```

| File | What it is |
| --- | --- |
| `warehouse.py` | The schema, a description of every table and column, and seeded data |
| `skills/metrics/SKILL.md` | The agreed definition of each metric |
| `agent.py` | The prompt and four tools: `list_tables`, `describe_table`, `run_sql`, `record_answer` |
| `env.py` | A fresh warehouse and an empty answer list per trial, and three graders |
| `tasks.yaml` | Eight questions, each with a reference query |

## What it takes to trust a number

A team will only use an analyst agent if its numbers can be trusted. So the design starts from
the ways a number goes wrong, and puts a defence against each one where it holds most firmly:

| What goes wrong | The defence | Where it lives |
| --- | --- | --- |
| The agent joins the wrong tables, or misreads a column | It reads the live schema and a description of every column before writing SQL | `describe_table` |
| It uses the wrong definition: counts refunds, includes test accounts | The agreed definitions are written down, and loaded before any calculation | `skills/metrics/SKILL.md` |
| It makes up a figure, or misquotes the one it found | The answer is recorded as SQL, and its value is computed by code from that SQL. A person can rerun it | `record_answer` |
| It gives a figure without saying what it counts | The prompt asks for the period and definition with every figure | `SYSTEM`, and a judge criterion |
| It answers a question the data can't answer | The prompt says to say so, and the definitions list what isn't measured | `SYSTEM`, the skill |
| It changes the data | The connection is read-only. SQLite refuses any write, whatever the prompt says | `connect()` |

Two of these are guaranteed by code: the data can't change, and a recorded answer is whatever
its SQL returns. The rest are what the evals measure.

## Where the agent's knowledge comes from

The agent needs three kinds of knowledge, and each comes from a different place for a reason:

- **The schema** comes from the database itself (`PRAGMA table_info`), so it can't go out of
  date when a column is added.
- **What the columns mean** comes from descriptions kept next to the schema in `warehouse.py`.
  In a real warehouse they'd come from your data catalog or dbt docs. Good descriptions are
  the difference between a right join and a plausible wrong one, so say what a row is, how
  tables join, and every value a coded column can take ("completed, cancelled or refunded").
- **What the metrics mean** comes from a skill. Definitions are business decisions, not facts
  about the data, and they grow over time. A skill keeps them out of the prompt until they're
  needed, and gives the team one place to change them.

## How the answers are graded

Every grader looks at outcomes: the recorded answer, which is what the team receives, and the
reply. None checks which tools the agent called or in what order. An agent that finds the
answer by a route nobody expected still passes, as it should.

Correctness is graded by code. A judge is never asked whether a number is right.

**`answer`** compares the recorded answer, the rows the team would see, with the result of the
task's `reference_sql` run on the same warehouse:

- Column names and column order don't matter. Text is compared without case; numbers within
  `tolerance` (default 0.01, a penny).
- Row order doesn't matter unless the task sets `ordered: true`, as a "top three" does.
- The number of rows must match, so a breakdown with a missing or extra group fails.

The task stores the reference *query*, not the expected number. So the expected value can't go
stale if the seed data changes, and anyone reading the task can see exactly what "right" means,
including which orders count.

**`reply_states_answer`** checks that the reply contains the recorded figure, at whatever
precision the reply uses: "£4,160.10", "4160.1" and "£4,160" all match 4160.1. That's what makes
the number a person reads the number the query produced.

**`no_answer`** passes if nothing was recorded: for a question the data can't answer, no figure
should reach the team.

**The judge** is used only for meaning: whether the reply says what the figure counts, whether
it draws the right conclusion from a comparison, whether it says the data can't answer.

## The traps in the data

The data is generated with the problems real warehouses have, so that a plausible query gets a
wrong answer. In October:

| Query | Revenue |
| --- | --- |
| Every order, test accounts included | £5,903.45 |
| Completed orders, but at list price instead of the price paid | £4,247.50 |
| Completed orders, no test accounts, price paid: **the agreed definition** | **£4,160.10** |

Each task's comment says which mistake it would catch.

## The tasks

| Task | Suite | What it tests |
| --- | --- | --- |
| `revenue_for_a_month` | regression | The definition: completed orders, no test accounts, price paid |
| `active_customers_in_a_month` | regression | Counting distinct customers, not orders |
| `revenue_by_region` | regression | A breakdown: a join to customers and a group per region |
| `order_value_by_channel` | regression | The grain: revenue per order, not per order line; and the right conclusion |
| `no_data_for_nps` | regression | Saying the data can't answer, instead of estimating |
| `will_not_delete_data` | regression | A request to change data: refused, and honestly explained |
| `top_products_by_profit` | capability | Four tables, cost and price, a relative date range, ordering |
| `repeat_rate_for_a_quarter` | capability | A rate whose numerator and denominator come from the same population |

## Adapting it to your warehouse

- **Access.** Give the agent a database role that can only read, and only the schemas it needs.
  Set a statement timeout (here, `MAX_SECONDS`). On a warehouse that bills by data scanned, add
  a cost limit too, such as BigQuery's maximum bytes billed.
- **The catalog.** Generate the table and column descriptions from wherever your team already
  keeps them, so there's one source of truth.
- **The definitions.** Start from the ones your dashboards already use. If an agent's answer
  and a dashboard disagree, the team will trust neither.
- **The eval environment.** A trial needs data that never changes, so its expected answers
  don't either. Use a small fixture database with your real schema and seeded rows that
  include your real traps, or a frozen snapshot. Never point evals at a live warehouse: the
  right answer would change every day.
- **The tasks.** Take questions from the team's channel or tickets. For each, write the
  reference query a careful analyst would write. That query is also documentation of what the
  team means by the question.
