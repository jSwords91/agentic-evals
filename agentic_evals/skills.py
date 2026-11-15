"""Skills: procedures the model loads only when it needs them.

A skill is a folder holding a SKILL.md with a short frontmatter header:

    ---
    name: scheduling-meetings
    description: The team's rules for arranging meetings. Use before creating or changing one.
    ---
    # Scheduling meetings
    ...the procedure...

C2 [SKILLS][CTX]: progressive disclosure. The system prompt lists each skill's name and
  description, a line apiece. The model calls `load_skill` to read the full text when a task
  needs it, so a dozen procedures cost a dozen lines of context until one is used.
T5 [SKILLS][SKILLDOC]: the same SKILL.md format as Agent Skills in Claude Code and the API, so a
  skill written here can move there unchanged. A good description says what the skill holds and
  when to use it, in the third person: it's what the model reads to decide whether to load it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .tools import Tool, ToolError

# [SKILLDOC]: the published rules for a skill's name and description. Checking them here keeps a
#   skill portable to Claude Code and the API, which enforce the same rules.
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
MAX_NAME, MAX_DESCRIPTION = 64, 1024


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


def load_skills(directory: str | Path) -> dict[str, Skill]:
    """Read every <directory>/*/SKILL.md, sorted by folder so the prompt is stable (C3)."""
    skills = {}
    for path in sorted(Path(directory).glob("*/SKILL.md")):
        _, header, body = path.read_text().split("---", 2)
        meta = yaml.safe_load(header)
        skill = Skill(name=meta["name"], description=meta["description"], body=body.strip())
        if not (NAME.fullmatch(skill.name) and len(skill.name) <= MAX_NAME):
            raise ValueError(f"{path}: name must be lowercase letters, digits and hyphens, "
                             f"at most {MAX_NAME} characters.")
        if not 0 < len(skill.description) <= MAX_DESCRIPTION:
            raise ValueError(f"{path}: description must be 1 to {MAX_DESCRIPTION} characters.")
        if skill.name in skills:
            raise ValueError(f"Two skills are called {skill.name!r}.")
        skills[skill.name] = skill
    if not skills:
        raise ValueError(f"No */SKILL.md files in {directory}.")
    return skills


def skills_prompt(skills: dict[str, Skill]) -> str:
    """The section appended to the system prompt."""
    lines = "\n".join(f"- {s.name}: {s.description}" for s in skills.values())
    return f"Skills hold detailed procedures and definitions. Load the ones that apply before you act:\n{lines}"


def load_skill_tool(skills: dict[str, Skill]) -> Tool:
    async def run(args: dict) -> str:
        skill = skills.get(args.get("name", ""))
        if skill is None:
            raise ToolError(f"No skill called {args.get('name')!r}. Choose from: {', '.join(skills)}.")
        return skill.body

    return Tool(
        name="load_skill",
        description="Load the full procedure for one of the skills listed in your instructions.",
        # Built by hand rather than with @tool because the enum comes from the skills folder.
        input_schema={"type": "object", "additionalProperties": False, "required": ["name"],
                      "properties": {"name": {"type": "string", "enum": list(skills)}}},
        run=run,
    )

