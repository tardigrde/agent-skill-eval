from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import yaml

from agent_skill_eval.models import AgentType

SKILL_PATHS = {
    AgentType.OPENCODE: [".opencode/skills"],
    AgentType.CLAUDE_CODE: [".claude/skills"],
    AgentType.CODEX: [".codex/skills"],
    AgentType.FAKE: [".fake/skills"],
}


class SkillInstaller:
    def __init__(self, skill_path: Path):
        self.skill_path = skill_path
        self.skill_name = skill_path.name
        self.skill_md = skill_path / "SKILL.md"
        if not self.skill_md.exists():
            raise FileNotFoundError(f"SKILL.md not found in {skill_path}")

    def frontmatter_problems(self) -> list[str]:
        """Return human-readable problems with the SKILL.md frontmatter.

        Empty list means the frontmatter is usable. Agents rely on ``name``
        and ``description`` to decide when to trigger a skill, so missing
        fields cause silent misbehavior rather than errors.
        """
        text = self.skill_md.read_text()
        if not text.startswith("---"):
            return ["SKILL.md has no YAML frontmatter block (must start with '---')"]
        parts = text.split("---", 2)
        if len(parts) < 3:
            return ["SKILL.md frontmatter is not closed with '---'"]
        try:
            data = yaml.safe_load(parts[1])
        except yaml.YAMLError as e:
            return [f"SKILL.md frontmatter is not valid YAML: {e}"]
        if not isinstance(data, dict):
            return ["SKILL.md frontmatter is not a YAML mapping"]
        problems = []
        for field in ("name", "description"):
            if not data.get(field):
                problems.append(f"SKILL.md frontmatter is missing '{field}'")
        if data.get("name") and data["name"] != self.skill_name:
            problems.append(
                f"SKILL.md frontmatter name '{data['name']}' does not match skill directory name '{self.skill_name}'"
            )
        return problems

    def content_hash(self) -> str:
        """Hash SKILL.md and bundled resources that are installed with it."""
        digest = hashlib.sha256()
        for path in self._content_files():
            rel = path.relative_to(self.skill_path).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    def size_metrics(self) -> dict:
        """Return compact size metadata for the installed skill payload.

        This intentionally excludes ``evals/`` and cache directories just like
        ``content_hash``: the metric is about what can affect agent behavior
        after the skill is installed, not development fixtures.
        """
        files = []
        total_bytes = 0
        total_words = 0
        for path in self._content_files():
            rel = path.relative_to(self.skill_path).as_posix()
            data = path.read_bytes()
            text = data.decode("utf-8", errors="ignore")
            words = len(re.findall(r"\b\S+\b", text))
            digest = hashlib.sha256(data).hexdigest()
            files.append({"path": rel, "bytes": len(data), "words": words, "sha256": digest})
            total_bytes += len(data)
            total_words += words

        skill_text = self.skill_md.read_text(encoding="utf-8")
        body_text = self._skill_body(skill_text)
        return {
            "files": len(files),
            "bytes": total_bytes,
            "words": total_words,
            "skill_md": {
                "bytes": len(skill_text.encode("utf-8")),
                "words": len(re.findall(r"\b\S+\b", skill_text)),
                "body_words": len(re.findall(r"\b\S+\b", body_text)),
            },
            "content_files": files,
        }

    @staticmethod
    def _skill_body(text: str) -> str:
        if not text.startswith("---"):
            return text
        parts = text.split("---", 2)
        if len(parts) < 3:
            return text
        return parts[2].strip()

    def _content_files(self) -> list[Path]:
        ignored_dirs = {"evals", "__pycache__", ".git"}
        files: list[Path] = []
        for path in sorted(self.skill_path.rglob("*")):
            if path.is_dir():
                continue
            if any(part in ignored_dirs for part in path.relative_to(self.skill_path).parts):
                continue
            files.append(path)
        return files

    def install(self, workspace: Path, agent_type: AgentType) -> Path:
        target_dirs = SKILL_PATHS[agent_type]
        installed_to = None

        for target_dir in target_dirs:
            dest_base = workspace / target_dir / self.skill_name
            dest_base.mkdir(parents=True, exist_ok=True)

            shutil.copy2(self.skill_md, dest_base / "SKILL.md")

            for item in self.skill_path.iterdir():
                if item.name == "SKILL.md":
                    continue
                if item.is_file():
                    shutil.copy2(item, dest_base / item.name)
                elif item.is_dir() and item.name not in ("evals", "__pycache__", ".git"):
                    shutil.copytree(item, dest_base / item.name, dirs_exist_ok=True)

            installed_to = dest_base

        if installed_to is None:
            raise ValueError(f"No skill paths configured for {agent_type}")

        return installed_to

    def uninstall(self, workspace: Path, agent_type: AgentType) -> None:
        for target_dir in SKILL_PATHS[agent_type]:
            dest_base = workspace / target_dir / self.skill_name
            if dest_base.exists():
                shutil.rmtree(dest_base)

            parent = dest_base.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
