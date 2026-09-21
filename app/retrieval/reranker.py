"""Cross-encoder reranker (optional; degrades gracefully if the model can't load)."""
from __future__ import annotations

from typing import Optional

from app.utils.logger import get_logger

log = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._failed = False

    @property
    def model(self):
        if self._model is None and not self._failed:
            try:
                from sentence_transformers import CrossEncoder

                log.info("Loading reranker %s", self.model_name)
                self._model = CrossEncoder(self.model_name, max_length=512)
            except Exception as exc:
                self._failed = True
                log.warning("Reranker unavailable (%s); continuing without it", exc)
        return self._model

    def score(self, query: str, docs: list[dict]) -> Optional[list[float]]:
        if not docs or self.model is None:
            return None
        pairs = [(query, f"{d['file_path']} {d.get('symbol', '')}\n{d['content'][:1800]}") for d in docs]
        try:
            return [float(s) for s in self.model.predict(pairs, batch_size=16, show_progress_bar=False)]
        except Exception as exc:
            log.warning("Reranking failed: %s", exc)
            return None


_CACHE: dict[str, CrossEncoderReranker] = {}


def get_reranker(model_name: str) -> Optional[CrossEncoderReranker]:
    if not model_name:
        return None
    if model_name not in _CACHE:
        _CACHE[model_name] = CrossEncoderReranker(model_name)
    return _CACHE[model_name]
