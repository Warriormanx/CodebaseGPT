from functools import lru_cache

from app.pipeline import CodebaseGPT


@lru_cache(maxsize=1)
def get_engine() -> CodebaseGPT:
    return CodebaseGPT()
