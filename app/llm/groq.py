"""Thin Groq chat wrapper (swap this module to change LLM provider)."""
from __future__ import annotations

import time

from app.utils.logger import get_logger

log = get_logger(__name__)


class GroqLLM:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to .env or the sidebar.")
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.model = model

    def chat(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 1800) -> str:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # network / rate limit
                last_exc = exc
                log.warning("Groq call failed (attempt %d): %s", attempt + 1, str(exc)[:200])
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Groq request failed: {last_exc}")
