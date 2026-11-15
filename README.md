# agentic-evals

A worked example of building a Claude agent and evaluating it with task-based evals.

There are four example agents, from ten lines to a full assistant. Start with the one closest to
the agent you want to evaluate. Each has its own README explaining its design.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv run pytest -q                                                    # 148 offline tests, free
uv run python -m agentic_evals check examples/analyst/tasks.yaml    # replay reference solutions, free

export ANTHROPIC_API_KEY=...
uv run python -m examples.analyst                                   # ask an example agent questions
uv run python -m agentic_evals run examples/analyst/tasks.yaml --trials 3
uv run python -m agentic_evals run examples/assistant/tasks.yaml --suite regression --trials 3 --gate 0.9
```

## The examples

| Example | The agent | Its tools touch | In an eval, each trial gets |
| --- | --- | --- | --- |
| [`minimal`](examples/minimal) | A writing assistant with a `word_count` tool | Nothing: the tool is a pure function | Nothing |
| [`helpdesk`](examples/helpdesk) | Triages IT tickets | The team's ticket database, and a status page another team runs | A fresh test database, and a status page with fixed answers |
| [`analyst`](examples/analyst) | Answers questions about sales by writing SQL | A sales warehouse, read-only | A fresh copy of a warehouse with seeded data |
| [`assistant`](examples/assistant) | Manages one person's email and calendar | Email, calendar and contacts, through an MCP server | A fresh fake workspace behind the same MCP server |

What each one teaches:

- **minimal**: the smallest working eval, and how to choose a grader for each fact.
- **helpdesk**: the shape of most real agents. The tools get their database and clients as
  arguments, so an eval can hand them test versions without changing the agent's code.
- **analyst**: how to make an agent's numbers trustworthy. The schema and column descriptions
  come from tools, the metric definitions from a skill, every answer is recorded as SQL, and
  each task is graded by running a reference query on the same data.
- **assistant**: everything together: MCP, skills, the injection screen, rules enforced in code,
  and tasks about prompt injection, privacy and impersonation.

Every example has the same four parts: the agent (`agent.py`), an environment for evals
(`env.py`), a tasks file (`tasks.yaml`), and `__main__.py` to chat with it. That's what you'd
write for your own agent.

## An agent in a few lines

```python
from agentic_evals import Agent, tool

@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())

agent = Agent(system="You are a concise writing assistant.", tools=[word_count])

async with agent.session() as session:
    reply = await session.send("How many words are in 'the quick brown fox jumps'?")
    print(reply.text, session.tool_log, session.cost_usd)
```

An `Agent` is the prompt, the model, the tools and the limits. A `Session` is one conversation:
the MCP connections, the message history, the log of tool calls and the cost. Each eval trial
gets a new session.

## How it fits together

```
                         Agent: system prompt, model, effort, limits
                           │
      ┌────────────────────┼─────────────────────────┬──────────────────────┐
      │                    │                         │                      │
  @tool functions     MCP servers               skills/*/SKILL.md      Screen (Haiku)
  run in-process      stdio, HTTP or            listed in the prompt,  checks text marked
                      in-process                loaded on demand       <untrusted> before
                                                                       the model reads it
      └────────────────────┴───────────┬─────────────┘
                                       │
                         Session: one loop, one conversation
                           calls the model, runs the tools it asks for (concurrently),
                           returns results in one message, stops at max_turns or max_cost

python -m agentic_evals run tasks.yaml
  for each task, for each trial:
    fresh Environment ─> agent.session().send(task.input)
                      ─> code graders read the final state, the tool calls and the reply
                      ─> an LLM judge reads the reply, one call per criterion
                      ─> transcript
  then: pass@k, pass^k, cost and latency per task, the gate, a diff against a baseline
```

## Building an agent

### Tools

`@tool` turns a typed Python function into a tool. The parameters become the input schema, and
the docstring becomes the description. The model sees only the name, the description and the
schema, so write the docstring for the model: what the tool does, when to use it, and anything
surprising about its arguments.

```python
@tool
def convert_time(
    time: Annotated[str, Field(description="Date and time, YYYY-MM-DD HH:MM")],
    from_zone: str,
    to_zone: str,
) -> str:
    """Convert a date and time from one time zone to another."""
```

Tools are sent with `strict: true`, so the model's arguments always match the schema. The
arguments are also validated with pydantic before your function runs, which covers rules strict
mode can't express, like a minimum value. Sync functions run in a worker thread.

Raise `ToolError` for an expected failure ("No ticket 999.", "SQL error: no such column"). The
model receives the message unchanged and can act on it. Any other exception is treated as a
bug, but the model still sees it as an error result and the conversation carries on.

### Tools that need a database or a client

Most real tools need something: a database connection, an API client, a config. Define them
inside a function that takes those things as arguments, and build the agent there:

```python
def build_agent(db: Path, status: StatusPage) -> Agent:
    @tool
    def check_service_status(service: Literal["vpn", "email", "wifi", "payroll"]) -> str:
        """The current status of a company service, from the status page."""
        return status.status(service)

    @tool
    def triage_ticket(ticket_id: int, priority: Priority, team: Team) -> str:
        """Set a ticket's priority and the team that will handle it."""
        ...  # writes to `db`

    return Agent(system=SYSTEM, tools=[check_service_status, triage_ticket, ...])
```

This is what makes an agent testable. Production calls `build_agent` with the real database and
the real status page. An eval calls it with a test database and a status page that gives fixed
answers. The agent's code is the same in both. See `examples/helpdesk/agent.py`.

### MCP servers

```python
Agent(..., mcp_servers=[stdio("python", "-m", "my_server", env={"DB": path})])
```

`mcp_servers` takes anything `mcp.Client` accepts: `stdio(...)` for a local command, a URL for a
remote server over Streamable HTTP, or an `MCPServer` object to run in-process in tests. MCP
tools look exactly like local tools to the loop and to the model.

Use MCP for systems the agent uses but doesn't own, such as an inbox, a calendar or another
team's service: the same server can then serve this agent, Claude Desktop or the next agent
someone builds. Use `@tool` for everything else.

### Skills

```python
Agent(..., skills="path/to/skills")    # a folder of */SKILL.md files
```

A skill is a procedure or a set of rules in a `SKILL.md` file, with a `name` and `description`
in its header. The system prompt lists each skill's name and description, and the model calls
`load_skill` to read the full text when it needs it. This is the same format as Agent Skills in
Claude Code and the API.

Put procedures and definitions in skills, rules that must always hold in tool code, and only
what the model needs on every turn in the system prompt. The analyst's metric definitions are a
skill; the helpdesk's short triage policy lives in its prompt.

### The injection screen

```python
Agent(..., screen=Screen())   # Haiku 5.5 at low effort by default
```

A tool marks text written by third parties with `untrusted(text)`: an email body, a web page, a
review. Before a tool result reaches the model, a small model checks each marked span and
removes any that tries to instruct an AI. If the screen can't run, the text is removed rather
than passed through. Removals are listed in the transcript with the screen's reason.

The screen is one layer. The marking itself, the system prompt and the rules in tool code are
the others. The screen has its own eval (see [The screen's eval](#the-screens-eval)), and
`--no-screen` on an eval run shows whether it still earns its place.

### Limits and stop reasons

The harness stops the loop at `max_turns` (default 20) or `max_cost_usd` (default $1.00).
`Reply.stop_reason` says why a turn ended: `end_turn` when the model finished, `max_turns` or
`max_cost` for the harness's limits, `refusal` or `max_tokens` for the API's. Requests to Opus
5.5 and Sonnet 5.5 turn on server-side refusal fallback.

## Evals

### The environment: what a trial runs against

In production, your agent's tools touch real things: your database, someone's inbox, a payment
API. An eval can't use those, for two reasons. A trial mustn't change anything real, and the
right answer has to be the same every time the task runs. So each trial runs the agent against a
fresh copy of whatever its tools touch, built from fixed data, and the graders read that copy
when it's done.

An `Environment` says how to make the copy and how to read it:

```python
class Helpdesk(Environment):
    def setup(self, workdir):        # make a fresh copy for one trial
        self.db = create_db(workdir / "tickets.db")

    def agent(self):                 # the agent, pointed at the copy
        return build_agent(self.db, FixedStatusPage(STATUS))

    def query(self, sql):            # read one value from the copy, for state graders
        ...

    def snapshot(self):              # optional: extra records to keep in the transcript
        ...
```

Every trial gets a new instance and an empty temporary folder, so no trial sees another's
changes. What the copy is depends on what your tools touch:

| Your tools... | In a trial, give them... | Example |
| --- | --- | --- |
| Are pure functions | Nothing: `setup` does nothing and there's no `query` | `minimal` |
| Read and write your own database | A fresh test database, created and seeded in `setup` | `helpdesk`: the tickets |
| Read a database you mustn't change | A fixture copy with your real schema and seeded rows, opened read-only | `analyst`: the warehouse |
| Read from a service you don't control | A fake that returns fixed answers, so a task can't pass on Monday and fail on Tuesday because the service changed | `helpdesk`: the status page |
| Change things in a system you don't own (email, a CRM, payments) | A fake with the same interface, or a sandbox account you reset between trials | `assistant`: the workspace |

`query` can run SQL, read a file or call a sandbox API, whatever fits. It returns one value,
which a `state` grader compares with its target.

### The tasks file

```yaml
environment: examples.helpdesk.env:Helpdesk

tasks:
  - id: access_request_goes_to_security
    suite: regression
    input: Please triage 103.
    graders:
      - name: security_high
        type: state
        query: "SELECT COUNT(*) FROM tickets WHERE id = 103 AND team = 'security' AND priority = 'high'"
        expect: "== 1"
      - type: judge
        dimensions:
          no_promise: The message to the requester doesn't say or imply they'll be given access.
    reference_solution:
      - [get_ticket, {ticket_id: 103}]
      - [triage_ticket, {ticket_id: 103, priority: high, team: security}]
```

| Field | Meaning |
| --- | --- |
| `environment` | The `Environment` subclass the tasks run in, as `module:ClassName` |
| `id` | Unique name; also the transcript's file name |
| `suite` | `regression` (should pass nearly every time; gates CI) or `capability` (hard; reported, never gates) |
| `input` | The user's message, written the way a real user would write it |
| `graders` | One or more. A trial passes only if every grade passes |
| `reference_solution` | Tool calls, `[name, {args}]`, that pass every code grader |
| `reference_reply` | A correct reply, so the `reply` graders can be tested too (needed only if the task has them) |

Every field is validated when the file loads. A misspelt grader type, an unknown field, a bad
operator, an invalid regular expression or a duplicate id fails straight away, rather than
quietly running fewer checks than you wrote.

### Choosing a grader

**Grade the outcome, not the route.** An outcome is what the agent left behind: what changed in
the environment, what it delivered, and what its reply says. The route is which tools it called
and in what order. Agents often find valid routes the task's author didn't think of, and should
pass when they do. A task that says "must call `check_service_status`" fails an agent that got
the right answer another way, and passes one that called the tool and then ignored it.

| To check that... | Use | For example |
| --- | --- | --- |
| Something was changed, or wasn't | `state`: a value read from the environment, compared with a target | The ticket's priority is high; nothing was sent |
| The reply contains a fact with a fixed form | `reply`: a regular expression, `must_match` or `must_not_match` | "10am"; a person's name; "36" |
| Anything else code can decide | A custom grader (see [Add a grader type](#add-a-grader-type)) | The recorded answer matches a reference query; the calendar is unchanged; the reply is under 10 words |
| The reply means something: declines, warns, draws the right conclusion | `judge`: criteria in plain English, one model call each | "The reply says it can't delete data" |

**If a check can be written as code, write it as code.** Code graders are free, instant, and give
the same answer every time. Asking a judge whether a reply contains "36", or whether a number is
right, is slower, costs money and is sometimes wrong. The judge is for meaning, where the same
point can be made in many ways, including ways a pattern would misread ("it isn't short
enough").

Write each judge criterion so that two people reading the same transcript would grade it the
same way, and put the right answer in it when you know it: "The reply says customers spend more
on the website", not "the reply draws the right conclusion". The judge sees the input, every
tool call with its arguments, and the final reply, so it can grade a message the agent sent
through a tool.

There's a fifth grader, `tool_calls`, for the rare case where a call is itself the harm, whatever
happens next: reading a customer's records after their identity check failed is a privacy breach
even if nothing is changed or said. None of the examples needs it. "A question changes nothing"
is an outcome, so it's checked on the environment, not by listing tools that mustn't be called.

Two things that look like route checks aren't needed at all:

- **Whether a helpful tool was used.** If it helps, the outcome shows it: the agent gets more
  answers right. The helpdesk's status page is tested by a pair of tickets whose right
  priorities differ only because of what the status page says.
- **Unnecessary steps.** Extra tool calls cost time and money, and the report shows model calls,
  seconds and cost for every task. They're a cost to watch, not a failure.

### Running

```bash
python -m agentic_evals check TASKS     # validate, then replay reference solutions and replies (no model, free)
python -m agentic_evals run TASKS       # run the evals
```

| Option | Default | |
| --- | --- | --- |
| `--suite` | `all` | `regression`, `capability` or `all` |
| `--task ID` | | Run only this task; repeat for more |
| `--trials` | 3 | Trials per task (n) |
| `--k` | `--trials` | k for pass@k and pass^k |
| `--model`, `--effort` | the agent's | Try another model or effort level without changing code |
| `--no-screen` | | Turn off the injection screen |
| `--gate` | | Exit 1 unless mean regression pass^k reaches this |
| `--compare FILE` | | An earlier `results/latest.json`: shows which tasks flipped, and cost and latency |
| `--results DIR` | `results` | Where the report and transcripts go |
| `--concurrency` | 4 | Trials at once |

The report has one row per task: passes out of n, pass@k, pass^k, partial credit, cost per
trial, model calls, failed tool calls, seconds and cache hit rate, plus which graders failed. It
flags a capability task that passed every trial (move it to regression) and any task that never
passed (more often a broken task than a hard one: check it before blaming the agent). Then it
prints each failed trial: which grader failed and why, how the turn ended, the start of the
reply, and the transcript to open.

**pass@k and pass^k.** Agents don't give the same answer every time, so each task runs several
times. pass@k is the chance that at least one of k tries succeeds. pass^k is the chance that all
k succeed. At 75% per trial, pass^3 is about 42%. An agent that acts for someone has to work
every time, so the gate uses pass^k.

**Infrastructure errors.** An API outage, a rate limit or an MCP server that fails to start
isn't the agent's fault. Those trials are left out of pass^k and reported separately. The
category is kept narrow: a bug in a grader or the harness stops the run instead, because counting
it as an outage would quietly shrink the sample. A gate with no regression tasks to check fails.

### Reading a transcript

Each trial writes `results/transcripts/<task>-<trial>.json`, summary first:

| Field | What it tells you |
| --- | --- |
| `passed`, `score`, `grades` | The verdict. Each grade has a `detail`: the value found, what was missing, or the judge's reasoning |
| `failed`, `unknown` | Which graders failed, and which the judge couldn't decide (read those yourself) |
| `reply`, `stop_reason` | What the user got, and why the turn ended |
| `model_calls`, `tool_errors`, `cost_usd`, `cache_hit_rate`, `seconds` | What it cost, and how many tool calls failed |
| `tool_log` | Every tool call: its input, its output and whether it errored |
| `redacted` | Text the screen removed, and why |
| `environment` | Whatever the environment's `snapshot()` returns, such as the final state of the tickets |
| `messages` | Exactly what went to and from the model, including a summary of its reasoning |

To work through a failure:

1. **Start with the grades.** Is the failure real? An ambiguous criterion or a wrong query is a
   grader bug, not an agent bug. You don't know whether your graders are fair until you've read
   some transcripts.
2. **Then the tool log.** Did it call the right tools, find the right data, recover from errors?
3. **Then the messages**, around the step that went wrong. Did it misread a procedure (fix the
   prompt or skill), lack information (fix a tool's output or description), or ignore something
   it had (the prompt or the model)?
4. **Fix it, and make sure a task covers it**, so it stays fixed.

### The screen's eval

```bash
python -m agentic_evals screen examples/assistant/screen_cases.yaml --trials 3
```

The screen is a single classification call, so it's tested directly against 20 labelled
snippets: 10 attacks it must remove and 10 harmless ones it must keep. The harmless ones include
requests and AI-related words on purpose, so a screen that matches on scary words fails. A case
counts as right only if it's right on every trial. The pass thresholds (`--min-catch`,
`--min-keep`) are a judgement about which mistake costs more in your setting.

## Making it yours

### Evaluate an agent you already have

Take an agent your team might build: one that answers questions about sales by querying a
warehouse. The [`analyst`](examples/analyst) example is that agent, and its README goes through
the design in full. In outline:

1. **List what would make its output wrong, or harmful.** For the analyst: wrong joins, the
   wrong metric definition, a made-up number, a number without its definition, an answer to a
   question the data can't answer, a change to the data.
2. **Decide where each defence lives.** Code where it must always hold (a read-only connection;
   the answer recorded as SQL whose value code computes). Tools and skills where the agent needs
   knowledge (column descriptions; metric definitions). The prompt for the rest.
3. **Pass in what the tools touch.** Write `build_agent(...)` so it takes the database, clients
   and config as arguments (see
   [Tools that need a database or a client](#tools-that-need-a-database-or-a-client)).
   Production calls it with the real things.
4. **Write an environment.** Use the table in
   [The environment](#the-environment-what-a-trial-runs-against) to decide what a trial gets
   instead. For the analyst, it's a small warehouse with the real schema and seeded rows that
   include the real traps: test accounts, refunds, discounted prices.
5. **Write tasks from real requests.** Take them from the team's channel, tickets or logs, in the
   words people used. For each, decide what a right answer is and pick the grader from
   [Choosing a grader](#choosing-a-grader). For the analyst, that's a reference query per
   question, compared by code. Pair each request the agent should act on with one it shouldn't:
   a question the data can't answer, a request to delete data.
6. **Check it for free:** `python -m agentic_evals check tasks.yaml`. A failure here means a task
   or a grader is wrong, not the agent.
7. **Run it:** `python -m agentic_evals run tasks.yaml --trials 3`, and read the transcripts of
   the failures before you trust the graders. Grow the suite to 20 to 50 tasks, drawn from real
   requests and real mistakes, then add `--gate 0.9` to CI.

### Add a task

1. **Seed what it needs.** If the task is about a record (a ticket, a customer, an email), add
   that record to the environment's seed data. Give each task its own records, so one task's
   graders can't be confused by another's data.
2. **Add it to the tasks file.** An `id`; the `input`, in the words a user would use; `suite:
   regression` if it should already pass or `capability` if it's a stretch; the graders; a
   `reference_solution`; and a `reference_reply` if it has `reply` graders.
3. **Run `check`**, then `run --task <id> --trials 5`, and read the transcripts.

### Add a tool, and tasks for it

Say the helpdesk agent gets a new tool, `assign_to_person(ticket_id, email)`.

1. **Write it** inside `build_agent`, and add it to the agent's `tools` list. If it needs a new
   dependency (another API client, say), add that as an argument to `build_agent`, and have the
   environment pass a fake.
2. **Make its effect readable.** If the tool changes something, make sure `query` can see the
   change. Here that's an `assignee` column in the tickets table.
3. **Add at least two tasks, graded on the outcome.** One where the ticket should end up
   assigned (`SELECT COUNT(*) FROM tickets WHERE id = 105 AND assignee = '...'`), and one where
   it shouldn't (`... AND assignee IS NULL`). Don't grade whether `assign_to_person` was called:
   the assignment is what matters.
4. **Run the whole suite**, not just the new tasks. A new tool changes the choices the model
   has, and so can change how it handles the old tasks.

If the new tool only reads (a lookup, a status check), it changes nothing you can query. Test it
with tasks whose right outcome depends on what it returns. The helpdesk's status page is the
worked example: `known_incident_is_normal` and `office_outage_is_urgent` are tickets that read
alike, and the right priorities differ only because of what the status page says. Both tasks
grade the priority. Neither checks the tool call.

### Add a grader type

When the built-in graders can't express a check, write one:

```python
@register
class ReplyWordLimit(Grader):
    type: Literal["reply_word_limit"] = "reply_word_limit"
    name: str
    max_words: int
    reads_reply: ClassVar[bool] = True   # so `check` tests it against the reference reply

    async def grade(self, trial: Trial) -> list[Grade]:
        words = len(trial.reply.split())
        return [Grade(self.name, words <= self.max_words, f"{words} words")]
```

`trial` holds the task, the reply, the tool log and the environment. Define the grader in your
environment's module, or import it there, so it's registered before the tasks file is read. Then
use it in the tasks file as `type: reply_word_limit`. The examples have three more:
`not_disclosed` (assistant), and `answer` and `reply_states_answer` (analyst).

## Reading order

| File | What it is |
| --- | --- |
| `examples/minimal/` | The smallest agent, and the smallest eval |
| `examples/helpdesk/` | Tools that get their database and clients as arguments; a faked service |
| `agentic_evals/tools.py` | `@tool` and `ToolError` |
| `agentic_evals/mcp.py` | MCP servers as tools |
| `agentic_evals/skills.py` | Skills and `load_skill` |
| `agentic_evals/screen.py` | `untrusted()` and the screen |
| `agentic_evals/agent.py` | `Agent`, `Session` and the loop |
| `agentic_evals/usage.py` | Tokens and cost |
| `agentic_evals/evals/environment.py` | The `Environment` base class |
| `agentic_evals/evals/tasks.py` | The tasks file |
| `agentic_evals/evals/graders.py` | The built-in graders, and how to add your own |
| `agentic_evals/evals/metrics.py` | pass@k, pass^k and the gate |
| `agentic_evals/evals/runner.py` | Trials, transcripts and infrastructure errors |
| `agentic_evals/evals/screening.py` | The screen's component eval |
| `agentic_evals/__main__.py` | The command line |
| `examples/analyst/` | Trustworthy numbers: schema from tools, definitions in a skill, answers graded against reference queries |
| `examples/assistant/` | MCP, skills, the screen, rules in code, and safety tasks |
| `tests/` | 148 offline tests |
| `.github/workflows/evals.yml` | Lint, tests and every example's reference solutions on each PR; the gated regression suites when a key is set; capability nightly |

## Design notes

Each decision in the code carries a comment with an ID and a reference key, for example:

```python
# S3 [CONTAIN]: a hard rule. Whatever the prompt, an email or a user says, the only way
#   out of the studio is a draft that Maya sends herself.
```

To find every decision in one area, or everything that draws on one source:

```bash
grep -rn "E[0-9]* \[" --include=*.py --include=*.yaml .
grep -rn "\[DE\]" --include=*.py --include=*.yaml .
```

**Agent architecture (A)**

| ID | Decision |
| --- | --- |
| A1 | One loop. Tool calls in a batch run concurrently and their results go back in one message |
| A2 | The harness enforces turn and cost limits; the model isn't asked to |
| A3 | Simplest design first: no router, planner or subagents. A separate model judges, never the agent itself |
| A4 | Definition (`Agent`) and conversation (`Session`) are separate |
| A5 | A tool that fails returns an error the model can see, not a crashed conversation |
| A6 | A refusal or cut-off response ends the turn, and any tool calls in it are answered so the history stays valid |

**Prompting and context (C)**

| ID | Decision |
| --- | --- |
| C1 | A short system prompt: who, what good looks like, and the rules needed everywhere. Procedures in skills, hard rules in code |
| C2 | Skills load on demand; the prompt carries one line per skill |
| C3 | Prompt caching with a stable prefix: tools sorted by name, nothing in the system prompt that changes per call |
| C4 | History is only appended to, never edited, which keeps the cache and the model's earlier thinking valid |

**Tools, MCP and skills (T)**

| ID | Decision |
| --- | --- |
| T1 | Tools grouped by what the agent is trying to do, not one per table |
| T2 | Descriptions written for the model; strict schemas for tools defined here |
| T3 | Preconditions and input matching inside the tool; refusals say what to do next |
| T4 | Compact results with a row cap, saying when rows were cut |
| T5 | Skills in the standard SKILL.md format |
| T6 | MCP for systems the agent uses but doesn't own; local tools for everything else |

**Security and containment (S)**

| ID | Decision |
| --- | --- |
| S1 | An MCP subprocess gets only the variables it's given, so the API key never reaches it |
| S2 | Third-party text is marked, and the marker is escaped so the text can't close it early |
| S3 | Hard limits in code, behind the probabilistic layers: no external email, no write access to the warehouse |
| S4 | Minimal capability: only the tools the job needs; no tool that grants access, no shell, files or network |
| S5 | Every server call is audited, including refusals |
| S6 | Tools annotated as read-only, writing or irreversible |
| S7 | A cheap model screens marked text before the main model reads it, and fails closed |

**Cost (K)**

| ID | Decision |
| --- | --- |
| K1 | Model and effort are configuration; trying another is a flag on an eval run |
| K2 | Cost per trial from the API's usage figures, cache reads and writes included |
| K3 | Cache hit rate in every report |
| K4 | The narrow screening job runs on the cheapest model |

**Evaluations (E)**

| ID | Decision |
| --- | --- |
| E1 | Shared vocabulary: task, trial, transcript, outcome, grader, harness |
| E2 | `state` graders check the outcome, which the agent can't fake |
| E3 | Grade the outcome, not the route: no example checks which tools were called |
| E4 | The judge for meaning only: one call per criterion, UNKNOWN allowed, structured output |
| E5 | pass@k and pass^k; the gate uses pass^k |
| E6 | Regression and capability suites; capability tasks graduate |
| E7 | Balanced tasks: should act and should not act |
| E8 | A fresh environment for every trial |
| E9 | Reference solutions and replies replayed with no model, to prove each task can be passed |
| E10 | Transcripts with the tool log, the system's own records and the model's messages; baseline comparison |
| E11 | Partial credit alongside pass/fail |
| E12 | A component eval for the screen, alongside the end-to-end tasks |
| E13 | Infrastructure errors counted separately, and kept narrow |
| E14 | Domain checks as custom graders, in code |

## Not included, and why

- **The SDK's tool runner.** `client.beta.messages.tool_runner` with `@beta_tool` would replace
  the loop in `agent.py`, and its hooks would cover the limits and the screen. The loop is
  written by hand here so every step is visible in one place. A production build could
  reasonably start with the runner.
- **Subagents.** One loop handles every task here, and nothing in the evals suggests the context
  is getting crowded. Add them when an eval shows the need.
- **Compaction and memory.** Conversations here are short. A long-running agent should use the
  API's server-side compaction rather than editing history by hand.
- **Streaming.** Responses here are short enough to wait for. A chat interface would stream.
- **Multi-turn tasks.** Each task is one user message. A task with a scripted follow-up from the
  user would test clarifying questions end to end.
- **Judge calibration.** Before a judge gates anything, label 20 to 50 transcripts yourself and
  check it agrees.
- **The other layers.** Automated evals are the first line of defence, not the only one. In
  production they sit alongside monitoring of live traffic, user feedback, A/B tests of
  significant changes, and someone reading a sample of real transcripts every week. No single
  layer catches every problem.
- **Tool search and code execution with MCP.** These pay off with dozens of tools or large
  results. The examples' handful of tools, returning a few lines each, don't need them.

## Limitations

- The live evals haven't been run yet, so there are no published pass rates or costs. The
  offline tests pass, and requests were checked against `anthropic` 2.6.0 and `mcp` 2.3.0.
- The screen checks text that tools mark with `untrusted()`. A third-party MCP server won't mark
  anything, so with one of those you'd screen whole results instead.
- MCP tools return text only. Images and other content types would need converting.
- Check the prices in `agentic_evals/usage.py` against the pricing page before quoting a cost.
- When a refusal falls back to another model, the reported cost counts only the attempt that
  answered, at the agent's model's prices. The API's `usage.iterations` has the exact figures.
- Brightleaf, Harbour & Finch, Northfield Logistics and everyone in them are fictional.

## References

| Key | Source |
| --- | --- |
| DE | [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) |
| BEA | [Building effective AI agents](https://www.anthropic.com/engineering/building-effective-agents) |
| CTX | [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) |
| TOOLS | [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents) |
| SKILLS | [Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) |
| CEM | [Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) |
| FOLD | [When to fold subagents into your main agent loop](https://claude.com/resources/articles/building-agents-when-to-fold-subagents-into-your-main-agent-loop) |
| HD | [Harness design for long-running application development](https://www.anthropic.com/research/harness-design-long-running-apps) |
| MA | [Scaling Managed Agents](https://www.anthropic.com/research/managed-agents) |
| CONTAIN | [How we contain Claude across products](https://www.anthropic.com/research/how-we-contain-claude) |
| SKILLDOC | [Skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices) |
| INJECT | [Mitigate jailbreaks and prompt injections](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks) |
| docs | Claude API docs: tool use, strict tools, structured outputs, prompt caching, effort, adaptive thinking, refusal fallbacks |

The agent's shape started from
[anthropics/claude-quickstarts/agents](https://github.com/anthropics/claude-quickstarts/tree/main/agents),
and the eval method from [eval-driven-agent](https://github.com/jSwords91/eval-driven-agent).

## License

[MIT](LICENSE)
