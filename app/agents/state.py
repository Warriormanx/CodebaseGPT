import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    # inputs
    repo_id: str
    question: str            # full instruction for the LLM
    search_query: str        # short text for retrieval + reranking
    history: list[dict]
    agentic: bool
    seed_queries: list[str]
    include_tree: bool
    repo_tree: list[str]
    top_k: int
    use_rerank: bool
    max_iterations: int
    # working state
    sub_queries: list[str]
    chunks: list[dict]
    iteration: int
    # outputs
    answer: str
    citations: list[dict]
    steps: Annotated[list[str], operator.add]
