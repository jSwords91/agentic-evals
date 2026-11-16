"""The walkthrough notebook: shipped without outputs, and its offline cells run.

It can't be run end to end here: most cells call the Claude API. So every cell must compile, and
every cell before the first API call is executed for real, which catches a notebook that's fallen
behind the library.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "improving-an-analyst-agent.ipynb"
CELLS = [c for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code"]
FLAGS = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT


def source(cell: dict) -> str:
    return "".join(cell["source"])


def test_the_notebook_is_shipped_without_outputs():
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in CELLS)


def test_every_code_cell_compiles():
    for cell in CELLS:
        compile(source(cell), "<cell>", "exec", flags=FLAGS)


def test_the_cells_before_the_first_api_call_run():
    first_call = next(i for i, c in enumerate(CELLS) if "Harness(" in source(c) or "await evaluate(" in source(c))
    namespace: dict = {}

    async def run_cells() -> None:
        for cell in CELLS[:first_call]:
            result = eval(compile(source(cell), "<cell>", "exec", flags=FLAGS), namespace)
            if asyncio.iscoroutine(result):
                await result

    asyncio.run(run_cells())
    assert len(namespace["suite"].tasks) == 6 and "V1" in namespace
