"""The separation of the brain, the hands and the session, checked rather than claimed (A4).

Each module may import only the modules listed for it. A change that, say, gives the hands a
reference to the brain fails here, and the failure names the import to remove.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "agentic_evals"

# The package's own modules each may import. The arrows point one way: tools at the bottom, the
# harness joining the three parts on top. Nothing here may import the evals (evals/, which build
# on all of it) or the examples.
ALLOWED = {
    "tools": set(),
    "usage": set(),
    "screen": set(),
    "mcp": {"tools"},
    "skills": {"tools"},
    "hands": {"tools", "mcp"},                  # the hands know nothing of the brain or the log
    "session": {"tools", "usage"},              # the log knows nothing of the brain or the hands
    "agent": {"screen", "usage"},               # the brain's config holds no tools and no state
    "harness": {"agent", "hands", "session", "screen", "skills", "tools"},
    "chat": {"agent", "hands", "harness", "session"},
}
CORE = set(ALLOWED)


def imports(path: Path) -> set[str]:
    """The agentic_evals modules a core module imports, by name (`evals` for anything in evals/)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 1:            # from .hands import Hands
            found |= {node.module.split(".")[0]} if node.module else {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("agentic_evals."):
            found.add(node.module.split(".")[1])
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "examples":
            found.add("examples")
        elif isinstance(node, ast.Import):
            found |= {a.name.split(".")[1] for a in node.names if a.name.startswith("agentic_evals.")}
    return found


@pytest.mark.parametrize("module", sorted(ALLOWED))
def test_each_core_module_imports_only_what_it_is_allowed(module):
    extra = imports(PACKAGE / f"{module}.py") - ALLOWED[module]
    assert not extra, f"agentic_evals/{module}.py must not import {', '.join(sorted(extra))}"


def test_every_core_module_has_a_declared_place():
    modules = {p.stem for p in PACKAGE.glob("*.py")} - {"__init__", "__main__"}
    assert modules == CORE, f"declare these in ALLOWED: {', '.join(sorted(modules - CORE))}"


@pytest.mark.parametrize("example", sorted(p.name for p in (ROOT / "examples").iterdir() if (p / "agent.py").exists()))
def test_an_example_brain_and_its_hands_do_not_know_each_other(example):
    folder = ROOT / "examples" / example

    def local(path: Path) -> set[str]:
        return {n.module.split(".")[0] for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module}

    assert not local(folder / "agent.py"), f"examples/{example}/agent.py must import only agentic_evals"
    assert not local(folder / "tools.py") & {"agent", "env"}, \
        f"examples/{example}/tools.py must not import the brain or the world"
