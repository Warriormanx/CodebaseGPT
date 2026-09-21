from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.deps import get_engine

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    repo_id: str
    question: str
    agentic: bool = False
    top_k: int = 8
    use_rerank: bool = True
    history: Optional[list[dict]] = None


@router.post("")
def chat(req: ChatRequest):
    engine = get_engine()
    if not engine.get_repo(req.repo_id):
        raise HTTPException(status_code=404, detail="Repository not ingested")
    try:
        return engine.ask(req.repo_id, req.question, req.history, req.agentic, req.top_k, req.use_rerank)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
