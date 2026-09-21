"""CodebaseGPT — Streamlit MVP.   Run:  streamlit run frontend/streamlit_app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app.pipeline import CodebaseGPT, with_overrides  # noqa: E402
from app.utils.config import settings  # noqa: E402

st.set_page_config(page_title="CodebaseGPT", page_icon="", layout="wide")

STATUS_ICON = {"verified": "✅", "file_only": "⚠️", "unverified": "❓"}
STATUS_HELP = {
    "verified": "Cited file and lines were part of the retrieved context",
    "file_only": "File was retrieved, but the cited lines were not in the context",
    "unverified": "File was not in the retrieved context — treat with caution",
}


@st.cache_resource(show_spinner="Connecting to MongoDB and loading models…")
def get_app(mongo_uri: str, groq_key: str, groq_model: str, gh_token: str) -> CodebaseGPT:
    cfg = with_overrides(settings, mongodb_uri=mongo_uri, groq_api_key=groq_key, groq_model=groq_model, github_token=gh_token)
    return CodebaseGPT(cfg)


def github_link(meta: dict, cit: dict) -> str:
    base = f"https://github.com/{meta.get('owner')}/{meta.get('name')}/blob/{meta.get('commit')}"
    return f"{base}/{cit['file_path']}#L{cit['start_line']}-L{cit['end_line']}"


def render_assistant_extras(msg: dict, meta: dict) -> None:
    citations = msg.get("citations") or []
    if citations:
        st.markdown("**Sources**")
        for cit in citations:
            icon = STATUS_ICON.get(cit["status"], "•")
            sym = f" · {cit['symbol']}" if cit.get("symbol") else ""
            label = f"{icon} {cit['file_path']}:{cit['start_line']}–{cit['end_line']}{sym}"
            with st.expander(label):
                st.caption(STATUS_HELP.get(cit["status"], ""))
                if meta.get("owner"):
                    st.markdown(f"[Open on GitHub ↗]({github_link(meta, cit)})")
                if cit.get("snippet"):
                    st.code(cit["snippet"], language=cit.get("language") or None, line_numbers=True)
    if msg.get("retrieved") and not citations:
        with st.expander("Retrieved context"):
            for r in msg["retrieved"]:
                st.markdown(f"- `{r['file_path']}:{r['start_line']}-{r['end_line']}` {r['kind']} {r['symbol']}")
    if msg.get("steps"):
        with st.expander("🔎 How this answer was produced"):
            for s in msg["steps"]:
                st.markdown(f"- {s}")


#  sidebar
with st.sidebar:
    st.title("CodebaseGPT")
    st.caption("Ask questions about any GitHub repo — grounded in file & line citations.")

    # with st.expander("⚙️ Connections", expanded=not settings.groq_api_key):
    #     mongo_uri = st.text_input("MongoDB URI", value=settings.mongodb_uri, type="password")
    #     groq_key = st.text_input("Groq API key", value=settings.groq_api_key, type="password")
    #     groq_model = st.text_input("Groq model", value=settings.groq_model)
    #     gh_token = st.text_input("GitHub token (private repos)", value=settings.github_token, type="password")
    
    with st.expander("⚙️ Connections", expanded=not (settings.groq_api_key and settings.mongodb_uri)):
        st.caption("Secrets from `.env` are used automatically and never displayed. "
                   "Type a value below only to override.")

        def _ph(val: str) -> str:
            return "using value from .env" if val else "not set"

        mongo_in = st.text_input("MongoDB URI (override)", value="", type="password",
                                 placeholder=_ph(settings.mongodb_uri), autocomplete="off")
        groq_in = st.text_input("Groq API key (override)", value="", type="password",
                                placeholder=_ph(settings.groq_api_key), autocomplete="off")
        gh_in = st.text_input("GitHub token (override, private repos)", value="", type="password",
                              placeholder=_ph(settings.github_token), autocomplete="off")
        groq_model = st.text_input("Groq model", value=settings.groq_model)  # not a secret

    # typed value wins, otherwise fall back to .env
    mongo_uri = mongo_in.strip() or settings.mongodb_uri
    groq_key = groq_in.strip() or settings.groq_api_key
    gh_token = gh_in.strip() or settings.github_token

    try:
        engine = get_app(mongo_uri, groq_key, groq_model, gh_token)
        engine.store.ping()
    except Exception as exc:
        st.error(f"Could not connect: {exc}")
        st.stop()

    mode = "Atlas Vector Search" if engine.store.atlas_ok else "local exact search (no Atlas)"
    st.caption(f"Vector backend: **{mode}**")

    st.divider()
    st.subheader("Index a repository")
    url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")
    branch = st.text_input("Branch (optional)")
    if st.button("Ingest repository", type="primary", disabled=not url.strip(), use_container_width=True):
        bar = st.progress(0.0, text="Starting…")
        try:
            meta = engine.ingest(url, branch or None, progress=lambda stage, f: bar.progress(min(max(f, 0.0), 1.0), text=stage))
            bar.empty()
            st.success(f"Indexed {meta['repo_id']}: {meta['files']} files → {meta['chunks']} chunks")
            if meta.get("truncated"):
                st.warning("Chunk limit reached — only part of the repository was indexed.")
            st.session_state["selected_repo"] = meta["repo_id"]
        except Exception as exc:
            bar.empty()
            st.error(f"Ingestion failed: {exc}")

    repos = engine.list_repos()
    repo_ids = [r["repo_id"] for r in repos]
    if repo_ids:
        st.divider()
        st.selectbox("Repository", repo_ids, key="selected_repo")

    st.divider()
    st.subheader("Answer settings")
    agentic = st.toggle("Agent mode (multi-step LangGraph)", value=False, help="Plans searches, checks whether the context is sufficient, and retrieves more if not.")
    use_rerank = st.toggle("Rerank results", value=bool(settings.reranker_model))
    top_k = st.slider("Chunks in context", 3, 15, 8)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("chats", None)
        st.rerun()

#  main
if not repo_ids:
    st.title("🧠 CodebaseGPT")
    st.info("Paste a GitHub URL in the sidebar and click **Ingest repository** to get started.")
    st.stop()

repo_id = st.session_state.get("selected_repo") or repo_ids[0]
meta = engine.get_repo(repo_id) or {}
st.title(repo_id)
langs = ", ".join(list((meta.get("languages") or {}).keys())[:5])
st.caption(f"{meta.get('files', '?')} files · {meta.get('chunks', '?')} chunks · branch `{meta.get('branch', '?')}` · "
           f"commit `{str(meta.get('commit', ''))[:7]}` · {langs}")

with st.expander("Slash commands"):
    st.markdown(
        "- `/explain authentication` — explain how a feature works\n"
        "- `/architecture` — high-level architecture overview\n"
        "- `/dependencies` — major dependencies and how they're used\n"
        "- `/find payment processing` — locate relevant code (no LLM call)\n"
        '- `/debug "NullPointerException in UserService"` — investigate an error\n'
        '- `/impact "What breaks if I change UserService?"` — trace usages'
    )

chats: dict = st.session_state.setdefault("chats", {})
messages: list[dict] = chats.setdefault(repo_id, [])

for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_assistant_extras(m, meta)

if prompt := st.chat_input("Ask about this repo… (try /architecture)"):
    history = [{"role": m["role"], "content": m["content"]} for m in messages[-6:]]
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the codebase…"):
                result = engine.ask(repo_id, prompt, history=history, agentic=agentic,
                                    top_k=top_k, use_rerank=use_rerank)
            msg = {"role": "assistant", "content": result["answer"], "citations": result["citations"],
                   "retrieved": result["retrieved"], "steps": result["steps"]}
            st.markdown(msg["content"])
            render_assistant_extras(msg, meta)
        except Exception as exc:
            msg = {"role": "assistant", "content": f"⚠️ Something went wrong: {exc}"}
            st.error(msg["content"])
    messages.append(msg)
