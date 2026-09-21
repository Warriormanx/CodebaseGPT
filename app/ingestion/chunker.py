"""Code-aware chunking.

Strategy
--------
* Code files  -> one chunk per function/class/method when it fits the size budget.
  Oversized classes are split into their methods (+ a "class_body" chunk for the
  header/fields); oversized leaf symbols fall back to overlapping line windows.
  Code *between* symbols (imports, constants, module-level statements) becomes
  "module" chunks so nothing is lost.
* Markdown    -> one chunk per heading section.
* Everything else (config, unsupported languages) -> overlapping line windows.

Every chunk carries: file_path, language, symbol, kind, start_line, end_line.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.ingestion.file_filter import language_kind
from app.ingestion.parser import Symbol, parse_symbols
from app.utils.config import settings

@dataclass
class ChunkConfig:
    max_lines: int = settings.max_chunk_lines
    max_chars: int = settings.max_chunk_chars
    overlap: int = settings.chunk_overlap_lines
    min_gap_chars: int = 30

@dataclass
class Chunk:
    file_path: str
    language: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    content: str

    @property
    def chunk_id(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def embedding_text(self) -> str:
        label = f"{self.kind} {self.symbol}".strip() if self.symbol else self.kind
        return f"File: {self.file_path}\nType: {label}\n\n{self.content}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "language": self.language,
            "symbol": self.symbol,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
        }


class _Emitter:
    def __init__(self, rel_path: str, language: str, lines: list[str], cfg: ChunkConfig):
        self.rel_path, self.language, self.lines, self.cfg = rel_path, language, lines, cfg
        self.chunks: list[Chunk] = []

    def _fits(self, start: int, end: int) -> bool:
        if end - start + 1 > self.cfg.max_lines:
            return False
        return sum(len(l) + 1 for l in self.lines[start - 1:end]) <= self.cfg.max_chars

    def _windows(self, start: int, end: int):
        i = start
        while i <= end:
            j, chars = i, 0
            while j <= end:
                n = len(self.lines[j - 1]) + 1
                if j > i and (j - i >= self.cfg.max_lines or chars + n > self.cfg.max_chars):
                    break
                chars += n
                j += 1
            yield i, j - 1
            if j > end:
                break
            i = max(j - self.cfg.overlap, i + 1)

    def emit(self, start: int, end: int, symbol: str, kind: str, gap: bool = False) -> None:
        while start <= end and not self.lines[start - 1].strip():
            start += 1
        while end >= start and not self.lines[end - 1].strip():
            end -= 1
        if start > end:
            return
        if gap and len("".join(self.lines[start - 1:end]).strip()) < self.cfg.min_gap_chars:
            return
        spans = [(start, end)] if self._fits(start, end) else list(self._windows(start, end))
        for s, e in spans:
            self.chunks.append(Chunk(self.rel_path, self.language, symbol, kind, s, e,
                                     "\n".join(self.lines[s - 1:e])))

    def region(self, symbols: list[Symbol], start: int, end: int, owner: Optional[Symbol]) -> None:
        gap_kind = "module" if owner is None else f"{owner.kind}_body"
        gap_name = "" if owner is None else owner.name
        cursor = start
        for sym in symbols:
            if sym.start > cursor:
                self.emit(cursor, sym.start - 1, gap_name, gap_kind, gap=True)
            if self._fits(sym.start, sym.end) or not sym.children:
                self.emit(sym.start, sym.end, sym.name, sym.kind)
            else:
                self.region(sym.children, sym.start, sym.end, sym)
            cursor = max(cursor, sym.end + 1)
        if cursor <= end:
            self.emit(cursor, end, gap_name, gap_kind, gap=True)


_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _chunk_markdown(em: _Emitter) -> None:
    starts: list[tuple[int, str]] = []
    in_fence = False
    for i, line in enumerate(em.lines, 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            starts.append((i, m.group(2)))
    if not starts or starts[0][0] > 1:
        starts.insert(0, (1, "(intro)"))
    for idx, (s, title) in enumerate(starts):
        e = starts[idx + 1][0] - 1 if idx + 1 < len(starts) else len(em.lines)
        em.emit(s, e, title, "doc")


def chunk_file(rel_path: str, source: str, language: str, cfg: Optional[ChunkConfig] = None) -> list[Chunk]:
    cfg = cfg or ChunkConfig()
    if not source.strip():
        return []
    lines = source.splitlines()
    em = _Emitter(rel_path, language, lines, cfg)
    kind = language_kind(language)

    if language == "markdown":
        _chunk_markdown(em)
    elif kind == "code":
        symbols = parse_symbols(source, language)
        if symbols:
            em.region(symbols, 1, len(lines), None)
        else:
            em.emit(1, len(lines), "", "module")
    else:
        em.emit(1, len(lines), "", kind)
    return em.chunks
