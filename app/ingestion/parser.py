"""Language-aware symbol extraction.

Python uses the real `ast` module. Other languages use lightweight regex
declaration matching + brace/`end` matching to find symbol boundaries. The result
is a tree of `Symbol`s (classes containing methods, etc.) with 1-indexed,
inclusive line ranges that the chunker turns into code-aware chunks.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Symbol:
    name: str
    kind: str  # class | function | method | interface | struct | ...
    start: int
    end: int
    children: list["Symbol"] = field(default_factory=list)


##### Python
def _parse_python(source: str) -> list[Symbol]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return []

    def start_of(node: ast.AST) -> int:
        decos = getattr(node, "decorator_list", [])
        return min([node.lineno] + [d.lineno for d in decos])

    def visit(body: list[ast.stmt], prefix: str, in_class: bool) -> list[Symbol]:
        out: list[Symbol] = []
        for node in body:
            if isinstance(node, ast.ClassDef):
                sym = Symbol(prefix + node.name, "class", start_of(node), node.end_lineno or node.lineno)
                sym.children = visit(node.body, sym.name + ".", True)
                out.append(sym)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if in_class else "function"
                out.append(Symbol(prefix + node.name, kind, start_of(node), node.end_lineno or node.lineno))
        return out

    return visit(tree.body, "", False)


#### regex-based languages
def _rx(p: str) -> re.Pattern:
    return re.compile(p)


_MODS = (r"(?:(?:public|private|protected|internal|static|final|abstract|synchronized|native|default|"
         r"override|virtual|async|sealed|partial|unsafe|extern|open|suspend)\s+)")

_JS = ("brace", [
    ("class", _rx(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)")),
    ("interface", _rx(r"^\s*(?:export\s+)?(?:interface|enum)\s+(\w+)")),
    ("function", _rx(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)")),
    ("function", _rx(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?=\s*"
                     r"(?:async\s+)?(?:function\b|\([^)]*\)\s*(?::\s*[^=]+)?=>|\w+\s*=>)")),
    ("method", _rx(r"^\s+(?:(?:public|private|protected|static|async|get|set)\s+)*(\w+)\s*\([^)]*\)"
                   r"\s*(?::\s*[^{]+)?\{\s*$")),
])
_GO = ("brace", [
    ("struct", _rx(r"^type\s+(\w+)\s+(?:struct|interface)\b")),
    ("function", _rx(r"^func\s+(?:\(\s*\w+\s+\*?(\w+)(?:\[[^\]]*\])?\s*\)\s*)?(\w+)")),
])
_RUST = ("brace", [
    ("struct", _rx(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+(\w+)")),
    ("impl", _rx(r"^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>, ]+\s+for\s+)?(\w+)")),
    ("function", _rx(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)")),
])
_JAVA = ("brace", [
    ("class", _rx(r"^\s*(?:@\w+\s+)*" + _MODS + r"*(?:class|interface|enum|record|object)\s+(\w+)")),
    ("function", _rx(r"^\s*(?:@\w+\s+)*" + _MODS + r"*fun\s+(?:<[^>]+>\s+)?(?:[\w.]+\.)?(\w+)\s*\(")),
    ("method", _rx(r"^\s+(?:@\w+(?:\([^)]*\))?\s+)*" + _MODS + r"+(?:<[^>]+>\s+)?"
                   r"(?:[\w<>\[\],.?& ]+?\s+)?(\w+)\s*\(")),
])
_C = ("brace", [
    ("class", _rx(r"^\s*(?:typedef\s+)?(?:class|struct)\s+(\w+)")),
    ("function", _rx(r"^[A-Za-z_][\w:<>,*&\s]*?[\s*&](\w+)\s*\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{?\s*$")),
])
_PHP = ("brace", [
    ("class", _rx(r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait)\s+(\w+)")),
    ("function", _rx(r"^\s*(?:(?:public|private|protected|static|final|abstract)\s+)*function\s+&?(\w+)")),
])
_SWIFT = ("brace", [
    ("class", _rx(r"^\s*(?:(?:public|private|internal|open|final)\s+)*(?:class|struct|enum|protocol|extension)\s+(\w+)")),
    ("function", _rx(r"^\s*(?:(?:public|private|internal|open|static|final|override)\s+)*func\s+(\w+)")),
])
_RUBY = ("ruby", [
    ("class", _rx(r"^\s*(?:class|module)\s+([\w:]+)")),
    ("function", _rx(r"^\s*def\s+(?:self\.)?(\w+[?!=]?)")),
])

LANG_SPECS = {
    "javascript": _JS, "typescript": _JS, "go": _GO, "rust": _RUST,
    "java": _JAVA, "csharp": _JAVA, "kotlin": _JAVA, "scala": _JAVA,
    "c": _C, "cpp": _C, "php": _PHP, "swift": _SWIFT, "ruby": _RUBY,
}
_NOT_NAMES = {"if", "for", "while", "switch", "return", "else", "catch", "sizeof", "do", "function",
              "elif", "try", "with", "new", "await", "typeof", "constructor_call"}
_STR_OR_COMMENT = re.compile(
    r'//.*$|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`'
)


def _brace_end(lines: list[str], start_idx: int) -> Optional[int]:
    depth, seen = 0, False
    for i in range(start_idx, min(len(lines), start_idx + 5000)):
        clean = _STR_OR_COMMENT.sub("", lines[i])
        for ch in clean:
            if ch == "{":
                depth += 1
                seen = True
            elif ch == "}":
                depth -= 1
        if seen and depth <= 0:
            return i + 1
        if not seen and clean.rstrip().endswith(";"):
            return i + 1  # declaration without a body
        if not seen and i - start_idx > 6:
            return None
    return None


def _ruby_end(lines: list[str], start_idx: int) -> Optional[int]:
    indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
    if re.search(r"\bend\s*$", lines[start_idx]) and "def" in lines[start_idx] and ";" in lines[start_idx]:
        return start_idx + 1
    for j in range(start_idx + 1, len(lines)):
        line = lines[j]
        if line.strip() == "end" and len(line) - len(line.lstrip()) == indent:
            return j + 1
    return None


def _nest(symbols: list[Symbol]) -> list[Symbol]:
    symbols.sort(key=lambda s: (s.start, -s.end))
    roots: list[Symbol] = []
    stack: list[Symbol] = []
    for s in symbols:
        while stack and not (s.start >= stack[-1].start and s.end <= stack[-1].end):
            stack.pop()
        if stack:
            parent = stack[-1]
            if s.kind == "function" and parent.kind in {"class", "struct", "impl", "interface"}:
                s.kind = "method"
            s.name = f"{parent.name}.{s.name}"
            parent.children.append(s)
        else:
            roots.append(s)
        stack.append(s)
    return roots


def _parse_regex(source: str, spec: tuple) -> list[Symbol]:
    style, patterns = spec
    lines = source.splitlines()
    symbols: list[Symbol] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "/*", "*", "#")):
            continue
        for kind, rx in patterns:
            m = rx.match(line)
            if not m:
                continue
            name = ".".join(g for g in m.groups() if g)
            if not name or name.split(".")[-1] in _NOT_NAMES:
                continue
            end = _ruby_end(lines, i) if style == "ruby" else _brace_end(lines, i)
            if end is not None and end >= i + 1:
                symbols.append(Symbol(name, kind, i + 1, end))
            break
    return _nest(symbols)


def parse_symbols(source: str, language: str) -> list[Symbol]:
    """Return the top-level symbol tree for a source file ([] if unsupported)."""
    try:
        if language == "python":
            return _parse_python(source)
        spec = LANG_SPECS.get(language)
        return _parse_regex(source, spec) if spec else []
    except Exception:
        return []
