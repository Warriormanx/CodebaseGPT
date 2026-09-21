"""High-level orchestration used by both the Streamlit UI and the FastAPI backend."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Callable, Optional

from app.agents.commands import parse_command
from app.agents.graph import build_graph
from app.agents.nodes import Deps
from app.embeddings.huggingface import get_embedder
from app.ingestion.chunker import ChunkConfig, chunk_file
from app.ingestion.file_filter import iter_source_files, read_text
from app.ingestion.github_loader import cleanup_clone, clone_repo
from app.llm.groq import GroqLLM
from app.retrieval.reranker import get_reranker
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import MongoVectorStore
from app.utils.citations import chunk_to_citation
from app.utils.config import Settings, settings as default_settings
from app.utils.logger import get_logger

log = get_logger(__name__)
ProgressFn = Callable[[str, float], None]


class CodebaseGPT:
    def __init__(self, cfg: Settings = default_settings, embedder=None, store=None, llm=None):
        self.cfg = cfg
        self.store = store or MongoVectorStore(
            cfg.mongodb_uri, cfg.mongodb_db, cfg.mongodb_collection,
            cfg.vector_index_name, cfg.embedding_dim)
        if store is None:
            self.store.ensure_indexes()
        self.embedder = embedder or get_embedder(cfg.embedding_model)
        self.retriever = HybridRetriever(self.store, self.embedder, get_reranker(cfg.reranker_model))
        self._llm = llm
        self._graph = None

    # LLM/graph
    @property
    def llm(self):
        if self._llm is None:
            self._llm = GroqLLM(self.cfg.groq_api_key, self.cfg.groq_model)
        return self._llm

    @property
    def graph(self):
        if self._graph is None:
            self._graph = build_graph(Deps(retriever=self.retriever, llm=self.llm))
        return self._graph

    # ingest
    def ingest(self, url: str, branch: Optional[str] = None,
               progress: Optional[ProgressFn] = None) -> dict:
        cb: ProgressFn = progress or (lambda stage, frac: None)
        cb("Cloning repository…", 0.02)
        info = clone_repo(url, self.cfg.clone_dir, branch, self.cfg.github_token or None)
        try:
            cb("Scanning files…", 0.12)
            files = list(iter_source_files(info.path, self.cfg.max_file_bytes, self.cfg.max_files))
            ccfg = ChunkConfig(self.cfg.max_chunk_lines, self.cfg.max_chunk_chars, self.cfg.chunk_overlap_lines)

            chunks, tree, langs = [], [], Counter()
            truncated = False
            for i, f in enumerate(files):
                text = read_text(f.path)
                if text is None:
                    continue
                file_chunks = chunk_file(f.rel_path, text, f.language, ccfg)
                if file_chunks:
                    chunks.extend(file_chunks)
                    tree.append(f.rel_path)
                    langs[f.language] += 1
                if len(chunks) >= self.cfg.max_chunks:
                    chunks, truncated = chunks[:self.cfg.max_chunks], True
                    log.warning("Chunk cap (%d) reached; remaining files skipped", self.cfg.max_chunks)
                    break
                if i % 25 == 0:
                    cb(f"Parsing & chunking ({i}/{len(files)} files)…", 0.12 + 0.25 * i / max(len(files), 1))
            if not chunks:
                raise ValueError("No indexable source files were found in this repository.")

            texts = [c.embedding_text for c in chunks]
            vectors = self.embedder.embed_documents(
                texts, lambda done, total: cb(f"Embedding chunks ({done}/{total})…", 0.4 + 0.5 * done / total))

            cb("Writing to MongoDB…", 0.93)
            docs = [{**c.to_dict(), "repo_id": info.repo_id, "embedding": v} for c, v in zip(chunks, vectors)]
            meta = {
                "url": info.url, "owner": info.owner, "name": info.name, "branch": info.branch,
                "commit": info.commit, "files": len(tree), "chunks": len(chunks),
                "languages": dict(langs.most_common()), "tree": tree[:500], "truncated": truncated,
            }
            self.store.replace_repo(info.repo_id, docs, meta)
            cb("Done", 1.0)
            return {k: v for k, v in meta.items() if k != "tree"} | {"repo_id": info.repo_id}
        finally:
            cleanup_clone(info.path)

    # repo mgmt
    def list_repos(self) -> list[dict]:
        return self.store.list_repos()

    def get_repo(self, repo_id: str) -> Optional[dict]:
        return self.store.get_repo(repo_id)

    def delete_repo(self, repo_id: str) -> None:
        self.store.delete_repo(repo_id)

    # ask
    def ask(self, repo_id: str, text: str, history: Optional[list[dict]] = None,
            agentic: bool = False, top_k: int = 8, use_rerank: bool = True) -> dict:
        cmd = parse_command(text)
        repo = self.get_repo(repo_id) or {}

        if cmd.retrieval_only:  # /find — pure retrieval, no LLM
            cands = self.retriever.search(repo_id, cmd.search_query, limit=max(top_k * 3, 20))
            chunks = (self.retriever.rerank(cmd.search_query, cands, top_k) if use_rerank else cands[:top_k])
            lines = [f"Top {len(chunks)} matches for **{cmd.search_query}**:"]
            lines += [f"- `{c['file_path']}:{c['start_line']}-{c['end_line']}` — {c['kind']} {c.get('symbol', '')}".rstrip()
                      for c in chunks]
            return {"answer": "\n".join(lines), "command": "find",
                    "citations": [chunk_to_citation(c) for c in chunks],
                    "retrieved": self._brief(chunks), "steps": ["Hybrid search + rerank (no LLM call)"]}

        state = {
            "repo_id": repo_id, "question": cmd.question, "search_query": cmd.search_query,
            "history": history or [], "agentic": cmd.agentic if cmd.agentic is not None else agentic,
            "seed_queries": cmd.seed_queries, "include_tree": cmd.include_tree,
            "repo_tree": repo.get("tree", []), "top_k": top_k, "use_rerank": use_rerank,
            "max_iterations": 2, "iteration": 0, "chunks": [], "sub_queries": [], "steps": [],
        }
        result = self.graph.invoke(state)
        return {"answer": result["answer"], "command": cmd.name, "citations": result["citations"],
                "retrieved": self._brief(result["chunks"]), "steps": result["steps"]}

    @staticmethod
    def _brief(chunks: list[dict]) -> list[dict]:
        return [{"file_path": c["file_path"], "start_line": c["start_line"], "end_line": c["end_line"],
                 "symbol": c.get("symbol", ""), "kind": c["kind"], "score": round(float(c.get("score", 0)), 4)}
                for c in chunks]


def with_overrides(cfg: Settings, **kwargs) -> Settings:
    return replace(cfg, **{k: v for k, v in kwargs.items() if v is not None})
