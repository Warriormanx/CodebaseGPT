"""Prompt templates and context formatting."""
from __future__ import annotations

ANSWER_SYSTEM = """You are CodebaseGPT, a senior software engineer answering questions about a GitHub repository.

Rules:
1. Use ONLY the code context provided below. Never invent files, functions, or behaviour.
2. Cite evidence inline using the exact format `path/to/file.py:START-END` (a single line may be `path:LINE`), using the file paths and line numbers shown in the context. Every factual claim about the code needs a citation.
3. If the context does not contain enough information, say what is missing instead of guessing.
4. Prefer a short direct answer first, then numbered steps with citations. For processes, end with a one-line flow such as: Login -> Token creation -> Middleware -> Endpoint.
5. Keep code quotes minimal; refer to line ranges instead of pasting large blocks."""

PLAN_SYSTEM = """You plan searches over a code repository. Given a question, output ONLY a JSON array (max 3 strings) of short search queries that together would find the code needed to answer it. Mix natural language with likely identifiers (function/class/file names, config keys). No prose."""

REFLECT_SYSTEM = """You judge whether retrieved code snippets are sufficient to answer a question about a repository. Output ONLY JSON: {"sufficient": true|false, "follow_up_queries": ["..."]}. If not sufficient, give at most 2 NEW short search queries (identifiers or concepts) for the missing pieces (e.g. callers, config, error handling, DB access). No prose."""


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        label = f"{c['kind']} {c['symbol']}".strip() if c.get("symbol") else c["kind"]
        numbered = "\n".join(
            f"{c['start_line'] + n:>5} | {line}" for n, line in enumerate(c["content"].split("\n"))
        )
        blocks.append(
            f"[{i}] {c['file_path']}:{c['start_line']}-{c['end_line']} ({label})\n{numbered}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(repo_id: str, question: str, chunks: list[dict],
                      repo_tree: list[str] | None = None) -> str:
    parts = [f"Repository: {repo_id}"]
    if repo_tree:
        parts.append("# File tree (truncated)\n" + "\n".join(repo_tree[:300]))
    parts.append("# Code context\n" + (build_context(chunks) or "(no relevant code retrieved)"))
    parts.append("# Question\n" + question)
    return "\n\n".join(parts)
