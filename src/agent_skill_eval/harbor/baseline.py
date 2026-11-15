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

from agent_skill_eval.git_state import safe_git_env
from agent_skill_eval.harbor.models import HarborBaseline, HarborBaselineFile

BASELINE_SHA_FILE = ".git/eval-baseline"
_INTERNAL_NAMES = {".eval-baseline", ".git"}
_COMMIT_ENV = {
    "GIT_AUTHOR_NAME": "agent-skill-eval",
    "GIT_AUTHOR_EMAIL": "eval@agent-skill-eval.local",
    "GIT_COMMITTER_NAME": "agent-skill-eval",
    "GIT_COMMITTER_EMAIL": "eval@agent-skill-eval.local",
    "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
    "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def _git_baseline(root: Path) -> tuple[Optional[str], Optional[str]]:
    """Create the same baseline commit as the task Dockerfile.

    Returns ``(tree_hash, commit_sha)``. The commit uses fixed identity and
    dates, so the SHA is reproducible across the exporter and the image.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="ase-baseline-") as temp:
            repo = Path(temp)
            for source in sorted(root.rglob("*")):
                if not source.is_file() or source.is_symlink():
                    continue
                target = repo / source.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            env = {**safe_git_env(repo), **_COMMIT_ENV}
            subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True, env=env)
            subprocess.run(
                ["git", "commit", "-q", "--allow-empty", "-m", "baseline"],
                cwd=repo,
                check=True,
                capture_output=True,
                env=env,
            )
            tree = subprocess.run(
                ["git", "rev-parse", "HEAD^{tree}"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.strip()
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.strip()
            return tree or None, commit or None
    except (OSError, subprocess.SubprocessError):
        return None, None


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
        tree_hash, commit_sha = _git_baseline(root)

    return HarborBaseline(
        workdir=workdir,
        tree_hash=tree_hash,
        commit_sha=commit_sha,
        files=files,
        dirs=sorted(dirs),
    )


def is_git_repo(workspace: Path) -> bool:
    git_dir = workspace / ".git"
    return git_dir.is_dir() and not git_dir.is_symlink()


def baseline_commit(workspace: Path) -> Optional[str]:
    """Return the baseline commit SHA recorded in the task image."""
    marker = workspace / BASELINE_SHA_FILE
    if marker.is_file() and not marker.is_symlink():
        try:
            resolved = marker.resolve()
            if resolved.is_relative_to(workspace.resolve()):
                value = resolved.read_text(encoding="utf-8").strip()
                if value:
                    return value
        except OSError:
            pass
    if not is_git_repo(workspace):
        return None
    try:
        result = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            env=safe_git_env(workspace),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    roots = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return roots[-1] if roots else None


def baseline_tree(workspace: Path, sha: str) -> Optional[str]:
    """Return the git tree hash of a commit, or None when it cannot be read."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", f"{sha}^{{tree}}"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            env=safe_git_env(workspace),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def baseline_commit_subject(workspace: Path, sha: str) -> str:
    """Return the short log line for the baseline commit."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%h %s", sha],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            env=safe_git_env(workspace),
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def workspace_files(root: Path) -> dict[str, tuple[str, int]]:
    """Hash every regular file under ``root``, ignoring git internals.

    Symlinks are skipped: following them would let an agent point a file at a
    host path and have the host-side grader read it.
    """
    files: dict[str, tuple[str, int]] = {}
    if not root.exists():
        return files
    root_resolved = root.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            if not path.resolve().is_relative_to(root_resolved):
                continue
        except OSError:
            continue
        relative = path.relative_to(root)
        if any(part in _INTERNAL_NAMES for part in relative.parts):
            continue
        files[relative.as_posix()] = (_sha256(path), _mode(path))
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
    """Return created, modified, deleted, and mode-changed paths."""
    expected = {item.path: (item.sha256, item.mode) for item in baseline.files if _matches(item.path, only)}
    actual = {path: value for path, value in workspace_files(workspace).items() if _matches(path, only)}

    violations: list[str] = []
    for path in sorted(set(actual) - set(expected)):
        violations.append(f"created: {path}")
    for path in sorted(set(expected) - set(actual)):
        violations.append(f"deleted: {path}")
    for path in sorted(set(expected) & set(actual)):
        if expected[path][0] != actual[path][0]:
            violations.append(f"modified: {path}")
        elif expected[path][1] != actual[path][1]:
            violations.append(f"mode changed: {path}")
    return violations
