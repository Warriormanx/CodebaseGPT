# 🤖 CodebaseGPT

### AI-Powered GitHub Repository Intelligence using RAG, Vector Search & Agentic AI

CodebaseGPT is an AI-powered codebase understanding and analysis system that allows developers to interact with entire GitHub repositories using natural language.

Instead of searching through hundreds of files manually, developers can ask questions such as:

> How does authentication work in this repository?

> Where is the recommendation model being called?

> Why could this endpoint return a 500 error?

> What files would be affected if I modify UserService?

CodebaseGPT uses Retrieval-Augmented Generation (RAG) to retrieve relevant source code and documentation before generating grounded answers. The system is designed to evolve into an agentic software-engineering assistant capable of multi-step repository investigation.

---

## ✨ Features

### 🔍 Repository Understanding
- Analyze an entire GitHub repository.
- Understand repository structure and source files.
- Support multiple programming languages.
- Extract useful metadata from files, classes, functions, and modules.

### 🧠 RAG-Based Code Search
- Semantic search over source code.
- Retrieve context relevant to natural-language questions.
- Store embeddings using MongoDB Atlas Vector Search.
- Use HuggingFace embedding models for semantic representation.

### 🔀 Hybrid Retrieval

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

This is useful for both semantic questions and exact identifiers such as `getUserById()`, `UserService`, `/api/recommend`, and `JWT_SECRET`.

### 📚 Source Citations

Answers are grounded in repository evidence.

Example:

```text
Authentication is implemented using JWT.

The token is generated in:
backend/auth/token.py
Lines 18–31

Token validation occurs in:
backend/middleware/auth.py
Lines 11–38
```

### 🤖 Agentic Repository Analysis

The planned LangGraph architecture supports multi-step investigations:

```text
User Question
     ↓
Find relevant endpoint
     ↓
Find associated service
     ↓
Find database interaction
     ↓
Inspect error handling
     ↓
Inspect configuration
     ↓
Generate grounded explanation
```

### 🐛 Debugging Assistance

CodebaseGPT can investigate problems such as:

```text
"Why is /recommend returning a 500 error?"
```

The agent can retrieve endpoint implementation, service-layer logic, database calls, configuration, exception handling, and related dependencies.

### 🏗️ Architecture Understanding

CodebaseGPT can generate a high-level representation of how a repository works.

```text
Frontend
   │
   ▼
API Layer
   │
   ▼
Service Layer
   │
   ▼
Database
```

---

# 🧩 Example Use Cases

### Understand a Repository
```text
How does authentication work?
```

### Locate Code
```text
Where is payment processing implemented?
```

### Explain a Function
```text
Explain the predict() function.
```

### Debug an Error
```text
Why could UserService throw a NullPointerException?
```

### Understand Architecture
```text
Explain the architecture of this repository.
```

### Impact Analysis
```text
What could break if I modify UserService?
```

### Dependency Analysis
```text
Which libraries are responsible for authentication?
```

---

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

---

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

---

# 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| RAG Framework | LangChain |
| Agent Framework | LangGraph |
| LLM | Groq |
| Embeddings | HuggingFace |
| Vector Database | MongoDB Atlas Vector Search |
| Frontend MVP | Streamlit |
| Frontend | React + Tailwind CSS |
| Repository Source | GitHub |
| Containerization | Docker |
| Deployment | AWS |
| Version Control | Git + GitHub |

---

# 📁 Project Structure

```text
CodebaseGPT/
│
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── routes_repo.py
│   │   └── routes_chat.py
│   ├── ingestion/
│   │   ├── github_loader.py
│   │   ├── file_filter.py
│   │   ├── parser.py
│   │   └── chunker.py
│   ├── embeddings/
│   │   └── huggingface.py
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── reranker.py
│   ├── agents/
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── state.py
│   ├── llm/
│   │   └── groq.py
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── frontend/
├── tests/
├── .env
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# 🚀 Development Roadmap

## Phase 1 — Basic RAG

```text
GitHub URL
    ↓
Clone Repository
    ↓
Read Files
    ↓
Chunk Code
    ↓
Generate Embeddings
    ↓
MongoDB Atlas
    ↓
Retrieve Context
    ↓
Groq LLM
    ↓
Generate Answer
```

## Phase 2 — Code-Aware RAG

Introduce code-specific chunking and metadata.

Example:

```json
{
  "repo": "my-project",
  "file": "backend/auth.py",
  "language": "python",
  "symbol": "authenticate_user",
  "start_line": 42,
  "end_line": 67,
  "type": "function"
}
```

This allows CodebaseGPT to retrieve meaningful code units instead of arbitrary text fragments.

## Phase 3 — Hybrid Retrieval

Implement:
- Semantic vector search
- Keyword search
- Hybrid retrieval
- Metadata filtering
- Reranking

```text
                Query
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Semantic Search       Keyword Search
        │                   │
        └─────────┬─────────┘
                  ▼
                Fusion
                  │
                  ▼
              Reranker
                  │
                  ▼
             Final Context
```

## Phase 4 — Citations

Every generated answer should contain references to the source code used.

Example:

```text
backend/auth/token.py:18-31
backend/middleware/auth.py:11-38
backend/routes/login.py:20-41
```

## Phase 5 — LangGraph Agent

Introduce multi-step reasoning and repository investigation.

```text
Question
   ↓
Planner
   ↓
Search relevant files
   ↓
Inspect dependencies
   ↓
Follow code relationships
   ↓
Retrieve additional context
   ↓
Validate evidence
   ↓
Generate answer
```

## Phase 6 — Advanced Repository Intelligence

Planned capabilities:

```text
/explain
/architecture
/dependencies
/find
/debug
/impact
```

Example:

```text
/impact UserService
```

Could identify:

```text
UserService
   │
   ├── AuthController
   ├── RecommendationService
   ├── OrderService
   └── UserRepository
```

---

# ⚙️ Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CodebaseGPT.git
cd CodebaseGPT
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
MONGODB_URI=your_mongodb_connection_string
GITHUB_TOKEN=your_github_token
```

Do not commit `.env` to GitHub.

Add it to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
```

---

# ▶️ Running the Backend

Start FastAPI:

```bash
uvicorn app.main:app --reload
```

API:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

# 🖥️ Running the Frontend

For the initial Streamlit version:

```bash
streamlit run frontend/app.py
```

# 🐳 Docker

Build the image:

```bash
docker build -t codebasegpt .
```

Run the container:

```bash
docker run --env-file .env -p 8000:8000 codebasegpt
```

For the complete multi-service setup:

```bash
docker compose up --build
```

---

# 🔐 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key for the LLM |
| `MONGODB_URI` | MongoDB Atlas connection string |
| `GITHUB_TOKEN` | GitHub API token |

Keep all secrets outside source control.

---

# 🧪 Example Questions

```text
How does authentication work?
```

```text
Where is the database connection initialized?
```

```text
Explain the recommendation pipeline.
```

```text
Which files handle API authentication?
```

```text
Where is JWT validation implemented?
```

```text
What happens when the /predict endpoint is called?
```

```text
Why might this function fail when the database is unavailable?
```

```text
What files depend on UserService?
```

---

# 🔬 RAG Pipeline

```text
                User Query
                    │
                    ▼
             Query Embedding
                    │
                    ▼
        MongoDB Vector Search
                    │
                    ▼
             Candidate Chunks
                    │
                    ▼
                Reranker
                    │
                    ▼
             Relevant Context
                    │
                    ▼
                 Groq LLM
                    │
                    ▼
          Grounded Response
                    │
                    ▼
               Citations
```

---

# 🎯 Why CodebaseGPT?

Traditional code search depends heavily on knowing:
- exact file names
- function names
- variable names
- keywords
- repository structure

CodebaseGPT aims to let developers interact with repositories using natural language and contextual reasoning.

Instead of:

```text
Search → Open file → Search again → Trace dependency
```

the goal becomes:

```text
Ask → Retrieve → Reason → Explain
```

---

# 🚧 Future Enhancements

- [ ] GitHub OAuth integration
- [ ] Private repository support
- [ ] Repository indexing dashboard
- [ ] Incremental repository updates
- [ ] AST-based code parsing
- [ ] Knowledge graph for code relationships
- [ ] Dependency graph visualization
- [ ] Automated architecture diagrams
- [ ] Pull request analysis
- [ ] Commit-aware RAG
- [ ] Git diff analysis
- [ ] Automated bug investigation
- [ ] Code change impact analysis
- [ ] Test generation
- [ ] Documentation generation
- [ ] Multi-repository search
- [ ] Repository comparison
- [ ] Long-term conversational memory

---

# 📊 Evaluation

CodebaseGPT should eventually be evaluated using:

### Retrieval
- Recall@K
- Precision@K
- MRR
- NDCG

### Generation
- Faithfulness
- Answer relevance
- Citation correctness
- Context utilization

### System
- Retrieval latency
- End-to-end latency
- Token usage
- Cost per query

---

# 🔒 Security Considerations

Because repositories may contain sensitive information, production deployments should consider:

- Secret detection before indexing
- Private repository authorization
- Access-control enforcement
- Secure API key storage
- Repository isolation
- Prompt-injection defenses
- Malicious-code handling
- Logging without exposing secrets

Never send private repository contents to an external LLM without appropriate authorization and security controls.

---

# 📌 Project Status

🚧 **Currently in development**

### Current target

```text
Phase 1
GitHub ingestion
      ↓
Code-aware chunking
      ↓
HuggingFace embeddings
      ↓
MongoDB Vector Search
      ↓
RAG retrieval
      ↓
Groq generation
```

The project will progressively evolve into an agentic AI software-engineering assistant.

---

# 👨‍💻 Author

**Aayush Raj**

B.Tech — Computer Science

Interested in:

```text
AI / ML
Generative AI
RAG
Agentic AI
Backend Development
Cloud
Web3
```

---

# ⭐ If You Like the Project

If CodebaseGPT helps you understand RAG, agentic AI, or code intelligence, consider giving the repository a ⭐.

---

## License

This project is currently intended for educational and portfolio purposes. Add an appropriate open-source license before distributing the project publicly.
