"""LangGraph workflow (with a plain-Python fallback if langgraph isn't installed).

        START ──agentic?──► plan ─► retrieve ─► reflect ─┐
          │                           ▲   ▲              │ needs more?
          └──────basic──────────────►─┘   └──────────────┘
                                      │ sufficient / max iterations
                                      ▼
                                    answer ─► END
"""
from __future__ import annotations

from app.agents.nodes import Deps, answer_node, plan_node, reflect_node, retrieve_node
from app.agents.state import AgentState


def _route_start(state: AgentState) -> str:
    return "plan_step" if state.get("agentic") else "retrieve_step"


def _route_after_retrieve(state: AgentState) -> str:
    return "reflect_step" if state.get("agentic") else "answer_step"


def _route_after_reflect(state: AgentState) -> str:
    more = state.get("sub_queries") and state.get("iteration", 0) < state.get("max_iterations", 2)
    return "retrieve_step" if more else "answer_step"


class _SequentialRunner:
    """Same control flow without langgraph."""

    def __init__(self, deps: Deps):
        self.deps = deps

    def invoke(self, state: AgentState) -> AgentState:
        st: dict = dict(state)
        st.setdefault("steps", [])

        def merge(update: dict) -> None:
            for k, v in update.items():
                st[k] = st["steps"] + v if k == "steps" else v

        if _route_start(st) == "plan_step":
            merge(plan_node(st, self.deps))
        while True:
            merge(retrieve_node(st, self.deps))
            if _route_after_retrieve(st) == "answer_step":
                break
            merge(reflect_node(st, self.deps))
            if _route_after_reflect(st) == "answer_step":
                break
        merge(answer_node(st, self.deps))
        return st  # type: ignore[return-value]


def build_graph(deps: Deps):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return _SequentialRunner(deps)

    g = StateGraph(AgentState)
    g.add_node("plan_step", lambda s: plan_node(s, deps))
    g.add_node("retrieve_step", lambda s: retrieve_node(s, deps))
    g.add_node("reflect_step", lambda s: reflect_node(s, deps))
    g.add_node("answer_step", lambda s: answer_node(s, deps))

    g.add_conditional_edges(START, _route_start, {"plan_step": "plan_step", "retrieve_step": "retrieve_step"})
    g.add_edge("plan_step", "retrieve_step")
    g.add_conditional_edges("retrieve_step", _route_after_retrieve,
                            {"reflect_step": "reflect_step", "answer_step": "answer_step"})
    g.add_conditional_edges("reflect_step", _route_after_reflect,
                            {"retrieve_step": "retrieve_step", "answer_step": "answer_step"})
    g.add_edge("answer_step", END)
    return g.compile()
