"""HuggingFace sentence-transformers embeddings (BAAI/bge-small-en-v1.5 by default)."""
from __future__ import annotations

from typing import Callable, Optional
from sentence_transformers import SentenceTransformer
from app.utils.logger import get_logger

log = get_logger(__name__)

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class HFEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32, device: Optional[str] = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:

            log.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = min(512, self._model.max_seq_length or 512)
        return self._model

    def embed_documents(self, texts: list[str], progress: Optional[Callable[[int, int], None]] = None) -> list[list[float]]:
        out: list[list[float]] = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            vecs = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            out.extend(v.astype("float32").tolist() for v in vecs)
            if progress:
                progress(min(i + self.batch_size, total), total)
        return out

    def embed_query(self, text: str) -> list[float]:
        if "bge" in self.model_name.lower():
            text = _QUERY_PREFIX + text
        vec = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)[0]
        return vec.astype("float32").tolist()


_CACHE: dict[str, HFEmbedder] = {}


def get_embedder(model_name: str) -> HFEmbedder:
    if model_name not in _CACHE:
        _CACHE[model_name] = HFEmbedder(model_name)
    return _CACHE[model_name]
