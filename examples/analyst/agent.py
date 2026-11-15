"""The analyst's brain: a prompt and the metric definitions.

    uv run python -m examples.analyst

The team has to be able to trust its numbers, so the design starts from what could make a
number wrong, and puts each defence where it holds most firmly:

  - Wrong tables or joins. The tools give the live schema and a description of every column
    (describe_table in tools.py), and the prompt says to read them before writing SQL.
  - Wrong definition (counting refunds, including test accounts). The agreed definitions are a
    skill, loaded before any calculation.
  - A number the model made up, or misremembered between query and reply. The answer is
    recorded as SQL (record_answer), and its value is computed by code from that SQL. A person
    can rerun it, and the evals grade it against a reference query.
  - Changing the data. The tools' connection is read-only, so no prompt can change it (S3).

The brain holds none of the tools or resources. They're joined for each conversation:

    async with ANALYST.session(warehouse_tools(db, answers)) as session: ...
"""

from pathlib import Path

from agentic_evals import Agent

HERE = Path(__file__).parent

SYSTEM = """You answer questions about sales for the commercial team at Brightleaf, a UK online shop selling houseplants, pots and plant care. It's Monday 16 November 2026.

Every figure you give must come from a query you ran on the warehouse. Look at the tables you need before you write SQL, and load the metric definitions before you calculate anything.

When you have the answer, record the query that produces it with record_answer. Then reply in a few sentences: the figure exactly as recorded, the period it covers, and what it counts.

If the data can't answer the question, say so and say what's missing. Never estimate a figure. You have read-only access to the warehouse."""



ANALYST = Agent(system=SYSTEM, skills=HERE / "skills")
