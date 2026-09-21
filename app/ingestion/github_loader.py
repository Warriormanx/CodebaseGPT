"""Clone a GitHub repository (shallow) and return metadata about it."""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.utils.logger import get_logger

log = get_logger(__name__)

_GITHUB_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/(?P<owner>[\w.-]+)/(?P<name>[\w.-]+?)"
    r"(?:\.git)?(?:/tree/(?P<branch>.+?))?/?$"
)


@dataclass
class RepoInfo:
    owner: str
    name: str
    url: str
    branch: str
    commit: str
    path: Path

    @property
    def repo_id(self) -> str:
        return f"{self.owner}/{self.name}"


def parse_repo_url(url: str) -> tuple[str, str, Optional[str]]:
    """Return (owner, name, branch|None). Accepts full URLs or 'owner/name'."""
    url = url.strip()
    if re.fullmatch(r"[\w.-]+/[\w.-]+", url):
        url = f"https://github.com/{url}"
    m = _GITHUB_RE.match(url)
    if not m:
        raise ValueError(f"Not a valid GitHub repository URL: {url!r}")
    return m["owner"], m["name"], m["branch"]


def _rmtree(path: Path) -> None:
    def handler(func, p, _exc):  # read-only files on Windows (.git objects)
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=handler)
    else:
        shutil.rmtree(path, onerror=handler)


def _git(args: list[str], cwd: Optional[Path] = None) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=600,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return out.stdout.strip()


def clone_repo(url: str, dest_root: str | Path, branch: Optional[str] = None,
               token: Optional[str] = None) -> RepoInfo:
    owner, name, url_branch = parse_repo_url(url)
    branch = branch or url_branch
    dest = Path(dest_root) / owner / name
    if dest.exists():
        _rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    auth = f"x-access-token:{token}@" if token else ""
    clone_url = f"https://{auth}github.com/{owner}/{name}.git"
    cmd = ["clone", "--depth", "1", "--single-branch"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [clone_url, str(dest)]

    log.info("Cloning %s/%s (branch=%s)", owner, name, branch or "default")
    try:
        _git(cmd)
    except FileNotFoundError as exc:
        raise RuntimeError("`git` is not installed or not on PATH.") from exc
    except RuntimeError as exc:
        msg = str(exc)
        if token:
            msg = msg.replace(token, "***")
        raise RuntimeError(f"git clone failed: {msg}") from None

    commit = _git(["rev-parse", "HEAD"], cwd=dest)
    actual_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=dest)
    return RepoInfo(owner, name, f"https://github.com/{owner}/{name}", actual_branch, commit, dest)


def cleanup_clone(path: Path) -> None:
    try:
        _rmtree(path)
    except Exception as exc:  # best effort
        log.warning("Could not remove %s: %s", path, exc)
