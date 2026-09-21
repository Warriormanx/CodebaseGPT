"""Optional FastAPI backend:  uvicorn app.main:app --reload"""
from fastapi import FastAPI

from app.api import routes_chat, routes_repo

app = FastAPI(title="CodebaseGPT API", version="0.1.0")
app.include_router(routes_repo.router)
app.include_router(routes_chat.router)


@app.get("/health")
def health():
    return {"status": "ok"}
