"""Extract `path:start-end` citations from an answer and verify them against retrieved chunks."""
from __future__ import annotations

import re

_CITE = re.compile(r"([\w@./\\-]+\.[A-Za-z0-9]{1,8}):(\d+)(?:\s*[-–—]\s*(\d+))?")


def _norm(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _matches_file(cited: str, actual: str) -> bool:
    cited, actual = _norm(cited), _norm(actual)
    return cited == actual or actual.endswith("/" + cited) or cited.endswith("/" + actual)


def slice_lines(chunks: list[dict], file_path: str, start: int, end: int) -> str:
    """Rebuild the cited line range from the retrieved chunks of that file."""
    lines: dict[int, str] = {}
    for c in chunks:
        if c["file_path"] != file_path:
            continue
        for offset, text in enumerate(c["content"].split("\n")):
            lines.setdefault(c["start_line"] + offset, text)
    picked = [lines[n] for n in range(start, end + 1) if n in lines]
    return "\n".join(picked)


def chunk_to_citation(c: dict, status: str = "verified") -> dict:
    return {
        "file_path": c["file_path"], "start_line": c["start_line"], "end_line": c["end_line"],
        "symbol": c.get("symbol", ""), "language": c.get("language", ""),
        "status": status, "snippet": c["content"],
    }


def extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """status: verified (file + lines were in retrieved context) | file_only | unverified"""
    out, seen = [], set()
    for m in _CITE.finditer(answer):
        cited_path, s = m.group(1), int(m.group(2))
        e = int(m.group(3)) if m.group(3) else s
        if e < s:
            s, e = e, s
        same_file = [c for c in chunks if _matches_file(cited_path, c["file_path"])]
        path = same_file[0]["file_path"] if same_file else _norm(cited_path)
        key = (path, s, e)
        if key in seen:
            continue
        seen.add(key)
        overlapping = [c for c in same_file if c["start_line"] <= e and c["end_line"] >= s]
        status = "verified" if overlapping else ("file_only" if same_file else "unverified")
        snippet = slice_lines(same_file, path, s, e) if overlapping else ""
        out.append({
            "file_path": path, "start_line": s, "end_line": e,
            "symbol": overlapping[0].get("symbol", "") if overlapping else "",
            "language": same_file[0].get("language", "") if same_file else "",
            "status": status, "snippet": snippet,
        })
    return out
