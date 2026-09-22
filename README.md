# CodebaseGPT

Agentic RAG for GitHub repositories: paste a repo URL, ask questions in plain English, get answers grounded in **file + line citations**.

```
GitHub repo → clone → file filter → code-aware chunker → HF embeddings → MongoDB Atlas Vector Search
                                                                              │
question → (LangGraph plan) → hybrid retrieval (vector + BM25, RRF) → rerank → Groq → answer + verified citations
```

## Features

### Repository Understanding
- Analyze an entire GitHub repository.
- Understand repository structure and source files.
- Support multiple programming languages.
- Extract useful metadata from files, classes, functions, and modules.

### RAG-Based Code Search
- Semantic search over source code.
- Retrieve context relevant to natural-language questions.
- Store embeddings using MongoDB Atlas Vector Search.
- Use HuggingFace embedding models for semantic representation.

### Hybrid Retrieval

```text
Semantic Search
       +
Keyword / Exact Search
       ↓
Hybrid Retrieval
       ↓
Reranking
       ↓
Relevant Context
```

# 🏗️ System Architecture

```text
                         ┌─────────────────┐
                         │  GitHub Repo    │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Repository      │
                         │ Ingestion       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Code Parser     │
                         │ + Chunker       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ HuggingFace     │
                         │ Embeddings      │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ MongoDB Atlas   │
                         │ Vector Search   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Hybrid          │
                         │ Retriever       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Reranker        │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Groq / LLM      │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Answer +        │
                         │ Citations       │
                         └─────────────────┘
```

# 🤖 Agentic Architecture

The advanced version introduces LangGraph:

```text
                         User Query
                              │
                              ▼
                         ┌─────────┐
                         │ Planner │
                         └────┬────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
       Code Search       Repo Search      Dependency Search
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                         ┌──────────┐
                         │ Reranker │
                         └────┬─────┘
                              │
                              ▼
                         ┌──────────┐
                         │   LLM    │
                         └────┬─────┘
                              │
                              ▼
                     Verified Answer
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env .env                                   # fill in MONGODB_URI and GROQ_API_KEY
streamlit run frontend/streamlit_app.py
```

Requirements: Python 3.10+, `git` on PATH, a free [Groq API key](https://console.groq.com), and MongoDB
(Atlas free tier recommended, or a local `mongod`). The first run downloads the embedding model (~130 MB).

**Atlas:** create a cluster, add your IP to the network access list, and paste the connection string into
`MONGODB_URI`. The app creates the `$vectorSearch` index itself (cosine, 384 dims, filter on `repo_id`);
it takes about a minute to become queryable. Until then — and on any non-Atlas MongoDB — the app
automatically uses an exact in-process cosine search, so everything works either way.
If you change `EMBEDDING_MODEL`, set `EMBEDDING_DIM` to match and drop the old index.

## Using it

1. Paste a URL (or `owner/repo`) in the sidebar → **Ingest repository**.
2. Ask a question, or use a command:

| Command | What it does |
|---|---|
| `/explain authentication` | Step-by-step explanation with citations |
| `/architecture` | High-level architecture overview (uses the file tree) |
| `/dependencies` | Major dependencies and where they're used |
| `/find payment processing` | Hybrid search only — no LLM call |
| `/debug "NullPointerException in UserService"` | Root-cause investigation |
| `/impact "What breaks if I change UserService?"` | Usage / blast-radius analysis |

**Agent mode** (sidebar toggle) runs the LangGraph loop: *plan searches → retrieve → reflect (is context enough?) → retrieve more → answer*. Slash commands other than `/find` always use it.

Citations are checked against what was actually retrieved: ✅ verified (file and lines were in context), ⚠️ file only, ❓ unverified (likely hallucinated path).

## Layout

```
app/
  ingestion/   github_loader.py  file_filter.py  parser.py  chunker.py
  embeddings/  huggingface.py
  retrieval/   vector_store.py  retriever.py  reranker.py
  agents/      state.py  nodes.py  graph.py  prompts.py  commands.py
  llm/         groq.py
  api/ main.py pipeline.py   (FastAPI backend + orchestration)
  utils/       config.py  logger.py  citations.py
frontend/streamlit_app.py
tests/
```

## Chunking

Python uses `ast`; JS/TS, Java, C#, Kotlin, Go, Rust, C/C++, PHP, Swift and Ruby use regex + brace/`end` matching.
Each function/class/method becomes one chunk (large classes split into methods), code between symbols becomes
`module` chunks, Markdown splits by heading, everything else uses overlapping line windows.
Each chunk stores `file_path, language, symbol, kind, start_line, end_line`.
To upgrade a language to a full parser, replace `parse_symbols` in `parser.py` with a tree-sitter implementation.

## Optional FastAPI backend

```bash
uvicorn app.main:app --reload
```

## Docker

```bash
docker compose up --build      # UI on :8501, API on :8000
```

## Good to go docker image of this repo at

```bash
https://hub.docker.com/repository/docker/warriormanx/codebasegpt/general
```

## Limits & notes

* Defaults: 2000 files, 8000 chunks, 300 KB per file; shallow files are indexed first.
* Ingesting a repo again replaces its previous index.
* Private repos: set `GITHUB_TOKEN` (or use the sidebar field).
* Local vector fallback loads a repo's embeddings into memory — fine for typical repos, use Atlas for very large ones.

# ⭐ If You Like the Project

If CodebaseGPT helps you understand RAG, agentic AI, or code intelligence, consider giving the repository a ⭐.
