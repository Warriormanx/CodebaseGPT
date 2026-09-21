"""MongoDB storage + vector search.

* On MongoDB Atlas the `$vectorSearch` index is created automatically.
* On a local/community MongoDB (no Atlas Search) it transparently falls back to an
  exact in-process cosine search, so the app works end-to-end in development.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from pymongo import MongoClient

from app.utils.logger import get_logger

log = get_logger(__name__)


class MongoVectorStore:
    def __init__(self, uri: str, db_name: str, collection: str, index_name: str, dims: int):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        self.db = self.client[db_name]
        self.chunks = self.db[collection]
        self.repos = self.db["repos"]
        self.index_name = index_name
        self.dims = dims
        self.atlas_ok: bool = False
        self._doc_cache: dict[str, list[dict]] = {}
        self._matrix_cache: dict[str, tuple[list[str], np.ndarray]] = {}

    # setup
    def ping(self) -> None:
        self.client.admin.command("ping")

    def ensure_indexes(self) -> None:
        self.chunks.create_index([("repo_id", 1)])
        self.chunks.create_index([("repo_id", 1), ("file_path", 1)])
        self.repos.create_index([("repo_id", 1)], unique=True)
        self._ensure_vector_index()

    def _ensure_vector_index(self) -> None:
        try:
            from pymongo.operations import SearchIndexModel

            if not list(self.chunks.list_search_indexes(self.index_name)):
                model = SearchIndexModel(
                    name=self.index_name,
                    type="vectorSearch",
                    definition={"fields": [
                        {"type": "vector", "path": "embedding",
                         "numDimensions": self.dims, "similarity": "cosine"},
                        {"type": "filter", "path": "repo_id"},
                    ]},
                )
                self.chunks.create_search_index(model=model)
                log.info("Created Atlas vector index '%s' (may take ~1 min to become ready)",
                         self.index_name)
            self.atlas_ok = True
        except Exception as exc:
            self.atlas_ok = False
            log.warning("Atlas Vector Search unavailable (%s). Using exact local cosine search.",
                        str(exc)[:160])

    # writes
    def replace_repo(self, repo_id: str, docs: list[dict], meta: dict) -> None:
        self.chunks.delete_many({"repo_id": repo_id})
        for i in range(0, len(docs), 500):
            self.chunks.insert_many(docs[i:i + 500], ordered=False)
        meta = {**meta, "repo_id": repo_id, "ingested_at": datetime.now(timezone.utc)}
        self.repos.replace_one({"repo_id": repo_id}, meta, upsert=True)
        self.invalidate(repo_id)

    def delete_repo(self, repo_id: str) -> None:
        self.chunks.delete_many({"repo_id": repo_id})
        self.repos.delete_one({"repo_id": repo_id})
        self.invalidate(repo_id)

    def invalidate(self, repo_id: str) -> None:
        self._doc_cache.pop(repo_id, None)
        self._matrix_cache.pop(repo_id, None)

    # reads
    def list_repos(self) -> list[dict]:
        return list(self.repos.find({}, {"_id": 0, "tree": 0}).sort("ingested_at", -1))

    def get_repo(self, repo_id: str) -> Optional[dict]:
        return self.repos.find_one({"repo_id": repo_id}, {"_id": 0})

    def get_docs(self, repo_id: str) -> list[dict]:
        """All chunks of a repo (without embeddings); cached — used for BM25."""
        if repo_id not in self._doc_cache:
            self._doc_cache[repo_id] = list(
                self.chunks.find({"repo_id": repo_id}, {"_id": 0, "embedding": 0})
            )
        return self._doc_cache[repo_id]

    def vector_search(self, repo_id: str, query_vec: list[float], k: int) -> list[dict]:
        if self.atlas_ok:
            try:
                pipeline: list[dict[str, Any]] = [
                    {"$vectorSearch": {
                        "index": self.index_name, "path": "embedding", "queryVector": query_vec,
                        "numCandidates": max(100, k * 10), "limit": k,
                        "filter": {"repo_id": repo_id},
                    }},
                    {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                    {"$project": {"_id": 0, "embedding": 0}},
                ]
                docs = list(self.chunks.aggregate(pipeline))
                if docs:
                    return docs
                log.info("Atlas returned no hits (index still building?) - using local search")
            except Exception as exc:
                log.warning("$vectorSearch failed (%s) - using local search", str(exc)[:160])
        return self._local_vector_search(repo_id, query_vec, k)

    def _local_vector_search(self, repo_id: str, query_vec: list[float], k: int) -> list[dict]:
        if repo_id not in self._matrix_cache:
            rows = list(self.chunks.find({"repo_id": repo_id}, {"_id": 0, "chunk_id": 1, "embedding": 1}))
            if not rows:
                return []
            ids = [r["chunk_id"] for r in rows]
            self._matrix_cache[repo_id] = (ids, np.asarray([r["embedding"] for r in rows], dtype=np.float32))
        ids, matrix = self._matrix_cache[repo_id]
        sims = matrix @ np.asarray(query_vec, dtype=np.float32)
        top = np.argsort(-sims)[:k]
        by_id = {d["chunk_id"]: d for d in self.get_docs(repo_id)}
        return [{**by_id[ids[i]], "score": float(sims[i])} for i in top if ids[i] in by_id]
