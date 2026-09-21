"""Slash commands: /explain /architecture /dependencies /find /debug /impact."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedCommand:
    name: Optional[str]
    question: str                      # instruction given to the LLM
    search_query: str = ""             # short text used for retrieval / reranking
    agentic: Optional[bool] = None     # None = respect the UI toggle
    seed_queries: list[str] = field(default_factory=list)
    include_tree: bool = False
    retrieval_only: bool = False


def parse_command(text: str) -> ParsedCommand:
    t = text.strip()
    if not t.startswith("/"):
        return ParsedCommand(None, t, t)
    cmd, _, arg = t.partition(" ")
    cmd, arg = cmd.lower(), arg.strip().strip("\"'")

    if cmd == "/find" and arg:
        return ParsedCommand("find", arg, arg, retrieval_only=True)
    if cmd == "/explain" and arg:
        return ParsedCommand(
            "explain",
            f"Explain how {arg} is implemented in this repository. Walk through the flow step by "
            f"step and cite the relevant files and line ranges.",
            arg, agentic=True, seed_queries=[arg])
    if cmd == "/architecture":
        return ParsedCommand(
            "architecture",
            "Give a high-level architecture overview of this repository: main components/modules, "
            "how they interact, entry points, data storage, and external services. Cite files.",
            "application entry point main setup routes database config architecture",
            agentic=True, include_tree=True,
            seed_queries=["README overview architecture", "application entry point main app setup",
                          "database models storage connection", "API routes endpoints handlers"])
    if cmd == "/dependencies":
        return ParsedCommand(
            "dependencies",
            "Identify the major dependencies of this repository (libraries, frameworks, services) "
            "and explain how each is used, citing the manifest files and the code that uses them.",
            "requirements package.json pyproject dependencies imports libraries",
            agentic=True, include_tree=True,
            seed_queries=["requirements dependencies package manifest", "import external library framework"])
    if cmd == "/debug" and arg:
        return ParsedCommand(
            "debug",
            f"Investigate this problem using the repository code: {arg}\nFind where it could originate "
            f"or be raised/handled, explain the likely root cause(s), and suggest a fix. Cite code.",
            arg, agentic=True, seed_queries=[arg, f"error handling exception {arg}"])
    if cmd == "/impact" and arg:
        return ParsedCommand(
            "impact",
            f"Impact analysis: {arg}\nLocate the definition and everything that uses or depends on it "
            f"(callers, imports, tests, config). Explain what would likely break or need updating. Cite code.",
            arg, agentic=True, seed_queries=[arg, f"usage of {arg} import call"])
    return ParsedCommand(None, t, t)
