"""Hybrid retriever: dense (vector) + sparse (BM25) search fused with RRF, then reranked."""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app.utils.logger import get_logger

log = get_logger(__name__)

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PARTS = re.compile(r"[A-Za-z][a-z0-9]+|[A-Z]+(?![a-z])|\d+")


def tokenize(text: str) -> list[str]:
    """Code-aware tokenizer: keeps identifiers whole AND splits camelCase / snake_case."""
    tokens: list[str] = []
    for word in _WORD.findall(text):
        tokens.append(word.lower())
        parts = _PARTS.findall(word)
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
    return [t for t in tokens if len(t) > 1]


def _doc_tokens(doc: dict) -> list[str]:
    head = tokenize(f"{doc['file_path']} {doc.get('symbol', '')}")
    return head * 2 + tokenize(doc["content"])  # boost path/symbol matches


class HybridRetriever:
    def __init__(self, store, embedder, reranker=None, rrf_k: int = 60):
        self.store, self.embedder, self.reranker, self.rrf_k = store, embedder, reranker, rrf_k
        self._bm25: dict[str, tuple[list[dict], BM25Okapi]] = {}

    ### sparse
    def _bm25_for(self, repo_id: str) -> Optional[tuple[list[dict], BM25Okapi]]:
        docs = self.store.get_docs(repo_id)
        cached = self._bm25.get(repo_id)
        if cached and cached[0] is docs:
            return cached
        if not docs:
            return None
        built = (docs, BM25Okapi([_doc_tokens(d) for d in docs]))
        self._bm25[repo_id] = built
        return built

    def keyword_search(self, repo_id: str, query: str, limit: int) -> list[dict]:
        built = self._bm25_for(repo_id)
        q = tokenize(query)
        if not built or not q:
            return []
        docs, bm25 = built
        scores = bm25.get_scores(q)
        out = []
        for i in np.argsort(-scores)[:limit]:
            if scores[i] > 0:
                out.append({**docs[i], "bm25_score": float(scores[i])})
        return out

    #### dense
    def semantic_search(self, repo_id: str, query: str, limit: int) -> list[dict]:
        hits = self.store.vector_search(repo_id, self.embedder.embed_query(query), limit)
        for h in hits:
            h["sem_score"] = h.pop("score", None)
        return hits

    ### hybrid
    def _rrf(self, ranked_lists: list[list[dict]], limit: int) -> list[dict]:
        fused: dict[str, float] = {}
        docs: dict[str, dict] = {}
        for lst in ranked_lists:
            for rank, d in enumerate(lst, 1):
                cid = d["chunk_id"]
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
                if cid in docs:
                    docs[cid].update({k: v for k, v in d.items() if k.endswith("_score")})
                else:
                    docs[cid] = dict(d)
        out = []
        for cid in sorted(fused, key=fused.get, reverse=True)[:limit]:
            docs[cid]["score"] = fused[cid]
            out.append(docs[cid])
        return out

    def search(self, repo_id: str, query: str, limit: int = 30) -> list[dict]:
        """Hybrid semantic + keyword search (no reranking)."""
        semantic = self.semantic_search(repo_id, query, limit)
        keyword = self.keyword_search(repo_id, query, limit)
        return self._rrf([semantic, keyword], limit)

    #### rerank
    def rerank(self, query: str, docs: list[dict], top_k: int) -> list[dict]:
        """Blend cross-encoder order with the fused order (robust for exact identifiers)."""
        docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)
        scores = self.reranker.score(query, docs) if self.reranker else None
        if scores is None:
            return docs[:top_k]
        rr_order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        rr_rank = {i: r for r, i in enumerate(rr_order, 1)}
        out = []
        for orig_rank, (i, d) in enumerate(enumerate(docs), 1):
            final = 1.0 / (self.rrf_k + orig_rank) + 1.0 / (self.rrf_k + rr_rank[i])
            out.append({**d, "rerank_score": scores[i], "final_score": final})
        out.sort(key=lambda d: d["final_score"], reverse=True)
        return out[:top_k]

    def retrieve(self, repo_id: str, query: str, top_k: int = 8, use_rerank: bool = True) -> list[dict]:
        candidates = self.search(repo_id, query, limit=max(top_k * 3, 20))
        if use_rerank:
            return self.rerank(query, candidates, top_k)
        return candidates[:top_k]
