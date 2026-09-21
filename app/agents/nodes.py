"""LangGraph nodes: plan -> retrieve -> (reflect -> retrieve)* -> answer."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from app.agents.prompts import ANSWER_SYSTEM, PLAN_SYSTEM, REFLECT_SYSTEM, build_user_prompt
from app.agents.state import AgentState
from app.utils.citations import extract_citations
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Deps:
    retriever: Any
    llm: Any


def _extract_json(text: str) -> Optional[Any]:
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


def _dedupe(items: list[str], limit: int) -> list[str]:
    seen, out = set(), []
    for x in items:
        x = (x or "").strip()
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out[:limit]


def _base_query(state: AgentState) -> str:
    return state.get("search_query") or state["question"]


def plan_node(state: AgentState, deps: Deps) -> dict:
    base = _base_query(state)
    planned: list[str] = []
    try:
        recent = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in (state.get("history") or [])[-4:])
        raw = deps.llm.chat(
            [{"role": "system", "content": PLAN_SYSTEM},
             {"role": "user", "content": f"Recent conversation:\n{recent}\n\nQuestion: {base}"}],
            temperature=0, max_tokens=250,
        )
        data = _extract_json(raw)
        if isinstance(data, list):
            planned = [str(x) for x in data]
        elif isinstance(data, dict):
            planned = [str(x) for x in data.get("queries", [])]
    except Exception as exc:
        log.warning("Planning failed: %s", exc)
    queries = _dedupe([base, *(state.get("seed_queries") or []), *planned], 5)
    return {"sub_queries": queries, "iteration": 0,
            "steps": [f"Plan: searching for {queries}"]}


def retrieve_node(state: AgentState, deps: Deps) -> dict:
    repo_id, base = state["repo_id"], _base_query(state)
    top_k = state.get("top_k", 8)
    if state.get("agentic"):
        queries = state.get("sub_queries") or [base]
    else:
        queries = _dedupe([base, *(state.get("seed_queries") or [])], 4)

    merged = {c["chunk_id"]: c for c in state.get("chunks") or []}
    for q in queries:
        for c in deps.retriever.search(repo_id, q, limit=max(top_k * 3, 20)):
            prev = merged.get(c["chunk_id"])
            if prev is None or c["score"] > prev.get("score", 0):
                merged[c["chunk_id"]] = c
    candidates = list(merged.values())
    if state.get("use_rerank", True):
        chunks = deps.retriever.rerank(base, candidates, top_k)
    else:
        chunks = sorted(candidates, key=lambda d: d.get("score", 0), reverse=True)[:top_k]
    return {"chunks": chunks,
            "steps": [f"Retrieved {len(candidates)} candidate chunks from {len(queries)} search(es); "
                      f"kept top {len(chunks)}"]}


def reflect_node(state: AgentState, deps: Deps) -> dict:
    iteration = state.get("iteration", 0) + 1
    if iteration >= state.get("max_iterations", 2):
        return {"iteration": iteration, "sub_queries": []}
    summary = "\n".join(
        f"- {c['file_path']}:{c['start_line']}-{c['end_line']} [{c['kind']} {c.get('symbol', '')}] "
        f"{c['content'][:160].strip().splitlines()[0] if c['content'].strip() else ''}"
        for c in state.get("chunks", [])
    )
    try:
        raw = deps.llm.chat(
            [{"role": "system", "content": REFLECT_SYSTEM},
             {"role": "user", "content": f"Question: {state['question']}\n\nRetrieved:\n{summary}"}],
            temperature=0, max_tokens=200,
        )
        data = _extract_json(raw)
    except Exception as exc:
        log.warning("Reflection failed: %s", exc)
        data = None
    if not isinstance(data, dict):
        data = {}
    follow = _dedupe([str(q) for q in data.get("follow_up_queries", [])], 2)
    if data.get("sufficient", True) or not follow:
        return {"iteration": iteration, "sub_queries": [],
                "steps": ["Reflect: retrieved context looks sufficient"]}
    return {"iteration": iteration, "sub_queries": follow,
            "steps": [f"Reflect: context incomplete - follow-up searches {follow}"]}


def answer_node(state: AgentState, deps: Deps) -> dict:
    chunks = state.get("chunks", [])
    tree = state.get("repo_tree") if state.get("include_tree") else None
    messages = [{"role": "system", "content": ANSWER_SYSTEM}]
    for m in (state.get("history") or [])[-6:]:
        messages.append({"role": m["role"], "content": m["content"][:1500]})
    messages.append({"role": "user",
                     "content": build_user_prompt(state["repo_id"], state["question"], chunks, tree)})
    answer = deps.llm.chat(messages, temperature=0.1, max_tokens=1800)
    citations = extract_citations(answer, chunks)
    ok = sum(c["status"] == "verified" for c in citations)
    return {"answer": answer, "citations": citations,
            "steps": [f"Answered with {len(citations)} citation(s), {ok} verified against retrieved code"]}
