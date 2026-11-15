"""Baseline snapshots for Harbor verifiers.

The task image commits its fixtures before the agent runs. The verifier gets a
copy of the workdir as an artifact, but it cannot see the host-side pre-run
snapshot. A baseline manifest makes file checks deterministic even when the
`.git` directory does not survive artifact transfer.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from agent_skill_eval.harbor.models import HarborBaseline, HarborBaselineFile

BASELINE_SHA_FILE = ".git/eval-baseline"
_INTERNAL_NAMES = {".eval-baseline", ".git"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mode(path: Path) -> int:
    return 0o755 if path.stat().st_mode & 0o111 else 0o644


def _git_tree_hash(root: Path) -> Optional[str]:
    """Compute the git tree hash of a directory without creating a commit."""
    try:
        with tempfile.TemporaryDirectory(prefix="ase-baseline-") as temp:
            repo = Path(temp)
            for source in sorted(root.rglob("*")):
                if not source.is_file():
                    continue
                target = repo / source.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
            result = subprocess.run(["git", "write-tree"], cwd=repo, check=True, capture_output=True, text=True)
            return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_baseline(fixture_map: dict[str, Path], workdir: str = "/app") -> HarborBaseline:
    """Build a baseline from workspace-relative target paths to source files."""
    files: list[HarborBaselineFile] = []
    dirs: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="ase-baseline-") as temp:
        root = Path(temp)
        for target, source in sorted(fixture_map.items()):
            destination = root / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            files.append(
                HarborBaselineFile(
                    path=Path(target).as_posix(),
                    sha256=_sha256(destination),
                    mode=_mode(destination),
                    size=destination.stat().st_size,
                )
            )
            for parent in Path(target).parents:
                if parent.as_posix() != ".":
                    dirs.add(parent.as_posix())
        tree_hash = _git_tree_hash(root) if files else None

    return HarborBaseline(
        workdir=workdir,
        tree_hash=tree_hash,
        files=files,
        dirs=sorted(dirs),
    )


def is_git_repo(workspace: Path) -> bool:
    return (workspace / ".git").is_dir()


def baseline_commit(workspace: Path) -> Optional[str]:
    """Return the baseline commit SHA recorded in the task image."""
    marker = workspace / BASELINE_SHA_FILE
    if marker.is_file():
        value = marker.read_text(encoding="utf-8").strip()
        if value:
            return value
    if not is_git_repo(workspace):
        return None
    try:
        result = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    roots = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return roots[-1] if roots else None


def baseline_commit_subject(workspace: Path, sha: str) -> str:
    """Return the short log line for the baseline commit."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%h %s", sha],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def workspace_files(root: Path) -> dict[str, str]:
    """Hash every regular file under ``root``, ignoring git internals."""
    files: dict[str, str] = {}
    if not root.exists():
        return files
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in _INTERNAL_NAMES for part in relative.parts):
            continue
        files[relative.as_posix()] = _sha256(path)
    return files


def _matches(path: str, only: Optional[str]) -> bool:
    if not only:
        return True
    only = Path(only).as_posix().rstrip("/")
    return path == only or path.startswith(only + "/")


def baseline_violations(
    workspace: Path,
    baseline: HarborBaseline,
    only: Optional[str] = None,
) -> list[str]:
    """Return created, modified, and deleted paths relative to the baseline."""
    expected = {item.path: item.sha256 for item in baseline.files if _matches(item.path, only)}
    actual = {path: digest for path, digest in workspace_files(workspace).items() if _matches(path, only)}

    violations: list[str] = []
    for path in sorted(set(actual) - set(expected)):
        violations.append(f"created: {path}")
    for path in sorted(set(expected) - set(actual)):
        violations.append(f"deleted: {path}")
    for path in sorted(set(expected) & set(actual)):
        if expected[path] != actual[path]:
            violations.append(f"modified: {path}")
    return violations
