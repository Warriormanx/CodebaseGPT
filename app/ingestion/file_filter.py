"""Decide which files in a repository are worth indexing."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

IGNORED_DIRS = {
    ".git", "node_modules", "dist", "build", "target", "out", "__pycache__", ".venv", "venv",
    "env", ".tox", ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "vendor",
    "bower_components", ".next", ".nuxt", "coverage", ".gradle", "Pods", "site-packages",
    ".terraform",
}
IGNORED_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Pipfile.lock",
    "composer.lock", "Cargo.lock", "go.sum", "uv.lock", ".DS_Store",
}
IGNORED_SUFFIXES = (".min.js", ".min.css", ".map", ".lock", ".svg", ".snap", ".ipynb")

EXT_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".cjs": "javascript", ".ts": "typescript", ".tsx": "typescript", ".java": "java",
    ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp",
    ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby", ".php": "php", ".kt": "kotlin",
    ".swift": "swift", ".scala": "scala", ".sh": "shell", ".bash": "shell", ".sql": "sql",
    ".md": "markdown", ".mdx": "markdown", ".rst": "rst", ".txt": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "ini",
    ".cfg": "ini", ".html": "html", ".css": "css", ".scss": "css", ".xml": "xml",
    ".gradle": "gradle", ".proto": "proto", ".tf": "terraform",
}
NAME_LANG = {"Dockerfile": "dockerfile", "Makefile": "makefile", "Jenkinsfile": "groovy"}

CODE_LANGS = {
    "python", "javascript", "typescript", "java", "go", "rust", "c", "cpp", "csharp", "ruby",
    "php", "kotlin", "swift", "scala", "shell", "sql", "groovy", "proto",
}
DOC_LANGS = {"markdown", "rst", "text"}


def language_kind(language: str) -> str:
    if language in CODE_LANGS:
        return "code"
    if language in DOC_LANGS:
        return "doc"
    return "config"


@dataclass
class SourceFile:
    path: Path
    rel_path: str
    language: str
    size: int


def detect_language(path: Path) -> Optional[str]:
    if path.name in NAME_LANG:
        return NAME_LANG[path.name]
    return EXT_LANG.get(path.suffix.lower())


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return b"\0" in fh.read(2048)
    except OSError:
        return True


def iter_source_files(root: Path, max_bytes: int = 300_000, max_files: int = 2000) -> Iterator[SourceFile]:
    root = Path(root)
    found: list[SourceFile] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in IGNORED_DIRS and not (d.startswith(".") and d != ".github")
        )
        for fname in filenames:
            if fname in IGNORED_FILES or fname.endswith(IGNORED_SUFFIXES):
                continue
            path = Path(dirpath) / fname
            lang = detect_language(path)
            if lang is None:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size == 0 or size > max_bytes or _is_binary(path):
                continue
            found.append(SourceFile(path, path.relative_to(root).as_posix(), lang, size))
    # shallow files first (READMEs, entry points) so the cap keeps the most useful ones
    found.sort(key=lambda f: (f.rel_path.count("/"), f.rel_path))
    yield from found[:max_files]


def read_text(path: Path) -> Optional[str]:
    """Read a file as text; None if it looks minified/generated."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if any(len(line) > 3000 for line in text.splitlines()):
        return None
    return text
