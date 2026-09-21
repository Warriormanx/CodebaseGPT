from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.deps import get_engine

router = APIRouter(prefix="/repos", tags=["repos"])


class IngestRequest(BaseModel):
    url: str
    branch: Optional[str] = None


@router.post("")
def ingest(req: IngestRequest):
    try:
        return get_engine().ingest(req.url, req.branch)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("")
def list_repos():
    return get_engine().list_repos()


@router.delete("/{repo_id:path}")
def delete_repo(repo_id: str):
    get_engine().delete_repo(repo_id)
    return {"deleted": repo_id}
