from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console

console = Console()


class WorkspaceManager:
    def __init__(self, base_dir: Path | None = None, source_repo: str | None = None):
        self.base_dir = base_dir or Path(tempfile.gettempdir())
        self.source_repo = source_repo
        self.workspaces: list[Path] = []

    def create_workspace(self, name: str, fixture_files: dict[str, Path] | None = None) -> Path:
        workspace = self.base_dir / f"agent-skill-eval-{name}"
        if workspace.exists():
            shutil.rmtree(workspace)

        if self.source_repo:
            self._clone_repo(workspace)
        else:
            workspace.mkdir(parents=True, exist_ok=True)
            self._init_git_repo(workspace)

        if fixture_files:
            self._copy_fixtures(workspace, fixture_files)

        # Fixtures belong to the baseline: leaving them untracked makes
        # file_unchanged / git_clean / side-effect contracts report the eval's
        # own inputs as agent edits. Fresh workspaces still get the historical
        # "Initial commit" (even without fixtures) so HEAD exists.
        if self.source_repo:
            self._commit_all(workspace, "Add eval fixtures")
        else:
            self._commit_all(workspace, "Initial commit", allow_empty=True)

        self.workspaces.append(workspace)
        return workspace

    @staticmethod
    def _copy_fixtures(workspace: Path, fixture_files: dict[str, Path]) -> None:
        for rel_path, src_path in fixture_files.items():
            dest = workspace / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src_path.is_dir():
                shutil.copytree(src_path, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dest)

    @staticmethod
    def _commit_all(workspace: Path, message: str, allow_empty: bool = False) -> bool:
        subprocess.run(["git", "add", "-A"], cwd=workspace, capture_output=True, check=True)
        if not allow_empty:
            staged = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=workspace,
                capture_output=True,
                check=False,
            )
            if staged.returncode == 0:
                return False
        commit_args = ["git", "commit", "-m", message]
        if allow_empty:
            commit_args.append("--allow-empty")
        subprocess.run(commit_args, cwd=workspace, capture_output=True, check=True)
        return True

    def commit_paths(self, workspace: Path, paths: list[Path], message: str) -> bool:
        """Commit specific paths (e.g. the installed skill) into the baseline.

        Returns False when the paths were already committed (no staged
        changes), so callers can avoid empty commits.
        """
        if not paths:
            return False
        subprocess.run(
            ["git", "add", "-A", "--", *(str(path) for path in paths)],
            cwd=workspace,
            capture_output=True,
            check=True,
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=workspace,
            capture_output=True,
            check=False,
        )
        if staged.returncode == 0:
            return False
        subprocess.run(["git", "commit", "-m", message], cwd=workspace, capture_output=True, check=True)
        return True

    def _clone_repo(self, workspace: Path) -> None:
        subprocess.run(
            ["git", "clone", self.source_repo, str(workspace)],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "eval@agent-skill-eval.local"],
            cwd=workspace,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Skill Eval"],
            cwd=workspace,
            capture_output=True,
            check=True,
        )

    def _init_git_repo(self, workspace: Path) -> None:
        subprocess.run(
            ["git", "init", "-b", "main"],
            cwd=workspace,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "eval@agent-skill-eval.local"],
            cwd=workspace,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Skill Eval"],
            cwd=workspace,
            capture_output=True,
            check=True,
        )

    def cleanup(self, workspace: Path) -> None:
        if workspace.exists():
            shutil.rmtree(workspace)
        if workspace in self.workspaces:
            self.workspaces.remove(workspace)

    def cleanup_all(self) -> None:
        for ws in self.workspaces[:]:
            self.cleanup(ws)
