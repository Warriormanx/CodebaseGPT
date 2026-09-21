"""Central configuration, loaded from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # MongoDB
    mongodb_uri: str = ""
    mongodb_db: str = "CodebaseGPT"
    mongodb_collection: str = "chunks"
    vector_index_name: str = "chunk_vector_index"

    # LLM
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # GitHub
    github_token: str = ""
    clone_dir: str = "data/repos"

    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking
    max_chunk_lines: int = 60
    max_chunk_chars: int = 2000
    chunk_overlap_lines: int = 5

    # Ingestion limits
    max_files: int = 2000
    max_chunks: int = 8000
    max_file_bytes: int = 300_000

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            mongodb_uri=os.getenv("MONGODB_URI", d.mongodb_uri),
            mongodb_db=os.getenv("MONGODB_DB", d.mongodb_db),
            mongodb_collection=os.getenv("MONGODB_COLLECTION", d.mongodb_collection),
            vector_index_name=os.getenv("VECTOR_INDEX_NAME", d.vector_index_name),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_model=os.getenv("GROQ_MODEL", d.groq_model),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            clone_dir=os.getenv("CLONE_DIR", d.clone_dir),
            embedding_model=os.getenv("EMBEDDING_MODEL", d.embedding_model),
            embedding_dim=_int("EMBEDDING_DIM", d.embedding_dim),
            reranker_model=os.getenv("RERANKER_MODEL", d.reranker_model),
            max_chunk_lines=_int("MAX_CHUNK_LINES", d.max_chunk_lines),
            max_chunk_chars=_int("MAX_CHUNK_CHARS", d.max_chunk_chars),
            chunk_overlap_lines=_int("CHUNK_OVERLAP_LINES", d.chunk_overlap_lines),
            max_files=_int("MAX_FILES", d.max_files),
            max_chunks=_int("MAX_CHUNKS", d.max_chunks),
            max_file_bytes=_int("MAX_FILE_BYTES", d.max_file_bytes),
        )


settings = Settings.from_env()
