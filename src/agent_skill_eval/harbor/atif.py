"""Read Harbor's ATIF trajectories for trigger, command, and access analysis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

_SKILL_TOOLS = {"skill", "activate_skill"}
_READ_TOOLS = {"read", "view", "open", "open_file", "read_file", "file_read", "cat"}
_SHELL_TOOLS = {"shell", "bash", "exec", "execute", "command", "command_execution", "run_command", "terminal"}
_SYSTEM_PREFIXES = (
    "/usr/",
    "/bin/",
    "/sbin/",
    "/lib/",
    "/lib64/",
    "/etc/",
    "/proc/",
    "/dev/",
    "/sys/",
    "/opt/",
    "/var/",
    "/root/.cache/",
    "/root/.local/",
    "/home/",
)
_ABS_PATH = re.compile(r"(?<![\w./-])/(?:[\w.@+-]+/)*[\w.@+-]+")


@dataclass
class ToolCall:
    step_id: int
    function_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    output: str = ""

    @property
    def name(self) -> str:
        return self.function_name.lower()

    @property
    def argument_text(self) -> str:
        try:
            return json.dumps(self.arguments, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(self.arguments)


def load_trajectory(trial_dir: Path) -> Optional[dict[str, Any]]:
    for candidate in (trial_dir / "agent" / "trajectory.json", trial_dir / "trajectory.json"):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
            return data if isinstance(data, dict) else None
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def tool_calls(trajectory: Optional[dict[str, Any]]) -> list[ToolCall]:
    if not trajectory:
        return []
    calls: list[ToolCall] = []
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            continue
        observations: dict[str, str] = {}
        observation = step.get("observation") or {}
        for result in observation.get("results") or []:
            if not isinstance(result, dict):
                continue
            call_id = str(result.get("source_call_id") or "")
            content = result.get("content")
            if isinstance(content, str):
                observations[call_id] = content
        for call in step.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("tool_call_id") or "")
            calls.append(
                ToolCall(
                    step_id=int(step.get("step_id") or 0),
                    function_name=str(call.get("function_name") or ""),
                    arguments=_as_dict(call.get("arguments")),
                    output=observations.get(call_id, ""),
                )
            )
    return calls


def _path_values(arguments: dict[str, Any]) -> Iterable[str]:
    for key in ("path", "file_path", "file", "filename", "target", "directory"):
        value = arguments.get(key)
        if isinstance(value, str):
            yield value


def detect_trigger(
    trajectory: Optional[dict[str, Any]],
    *,
    skill_names: Iterable[str],
) -> tuple[Optional[bool], str]:
    """Return (triggered, evidence). None means insufficient telemetry."""
    calls = tool_calls(trajectory)
    if not calls:
        return None, "ATIF has no tool calls"

    names = [name.lower() for name in skill_names if name]
    for call in calls:
        arg_text = call.argument_text.lower()
        output = call.output.lower()
        if call.name in _SKILL_TOOLS:
            failed = any(marker in output for marker in ("error", "not found", "unavailable", "failed"))
            if failed and "<skill_content" not in output:
                continue
            if "<skill_content" in output:
                if not names or any(name in output for name in names):
                    return True, f"{call.function_name} returned skill content"
            if any(name in arg_text for name in names):
                return True, f"{call.function_name} argument named the skill"
            if "skill.md" in arg_text:
                return True, f"{call.function_name} referenced SKILL.md"
        if "skill.md" in arg_text:
            return True, f"{call.function_name} read SKILL.md"
        if "/skills/" in arg_text and (not names or any(name in arg_text for name in names)):
            return True, f"{call.function_name} accessed the registered skills directory"
        if call.name in _READ_TOOLS:
            for path in _path_values(call.arguments):
                lowered = path.lower()
                if "skill.md" in lowered or (
                    "/skills/" in lowered and (not names or any(name in lowered for name in names))
                ):
                    return True, f"{call.function_name} read {path}"

    return False, "ATIF tool calls present, but none loaded the skill"


def executed_commands(trajectory: Optional[dict[str, Any]]) -> list[str]:
    commands: list[str] = []
    for call in tool_calls(trajectory):
        if call.name not in _SHELL_TOOLS:
            continue
        value = call.arguments.get("command") or call.arguments.get("cmd") or call.arguments.get("command_line")
        if isinstance(value, list):
            value = " ".join(str(part) for part in value)
        if isinstance(value, str) and value.strip():
            commands.append(value.strip())
    return commands


def transcript(trajectory: Optional[dict[str, Any]]) -> str:
    if not trajectory:
        return ""
    lines: list[str] = []
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            continue
        source = step.get("source") or "agent"
        message = step.get("message")
        if isinstance(message, str) and message:
            lines.append(f"[{source}] {message}")
        reasoning = step.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            lines.append(f"[{source} reasoning] {reasoning}")
        for call in step.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name = call.get("function_name") or ""
            arguments = call.get("arguments")
            lines.append(f"[tool {name}] {json.dumps(arguments, ensure_ascii=False, sort_keys=True)}")
        observation = step.get("observation") or {}
        for result in observation.get("results") or []:
            if isinstance(result, dict) and isinstance(result.get("content"), str):
                lines.append(f"[observation] {result['content']}")
    return "\n".join(lines)


def reasoning_tokens(trajectory: Optional[dict[str, Any]]) -> Optional[int]:
    if not trajectory:
        return None
    total = 0
    found = False
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            continue
        metrics = step.get("metrics") or {}
        extra = metrics.get("extra") or {}
        value = extra.get("reasoning_tokens") if isinstance(extra, dict) else None
        if value is None:
            value = metrics.get("reasoning_tokens")
        if value is not None:
            found = True
            total += int(value)
    return total if found else None


def out_of_workspace_access(
    trajectory: Optional[dict[str, Any]],
    *,
    workdir: str,
    extra_allowed_prefixes: Iterable[str] = (),
) -> list[str]:
    """Return paths outside the workdir that the agent read or named."""
    if not trajectory:
        return []
    workdir = workdir.rstrip("/") or "/"
    allowed = tuple(extra_allowed_prefixes)
    found: dict[str, None] = {}

    def record(path: str, tool: str) -> None:
        if not path.startswith("/"):
            return
        normalized = path.rstrip("/") or "/"
        if normalized == workdir or normalized.startswith(workdir + "/"):
            return
        if any(normalized.startswith(prefix) for prefix in allowed):
            return
        if normalized.startswith(_SYSTEM_PREFIXES):
            return
        found[f"{tool}:{normalized}"] = None

    for call in tool_calls(trajectory):
        if call.name in _READ_TOOLS:
            for path in _path_values(call.arguments):
                record(path, call.function_name)
        if call.name in _SHELL_TOOLS:
            command = call.arguments.get("command") or call.arguments.get("cmd")
            if isinstance(command, str):
                for match in _ABS_PATH.finditer(command):
                    record(match.group(0), call.function_name)
    return sorted(found)


def raw_agent_text(trial_dir: Path, *, limit_bytes: int = 2_000_000) -> str:
    """Concatenate the agent's own log files, skipping ATIF and binaries."""
    agent_dir = trial_dir / "agent"
    if not agent_dir.is_dir():
        return ""
    preferred = ["opencode-v2.txt", "opencode.txt", "claude-code.txt", "codex.txt"]
    candidates = [agent_dir / name for name in preferred if (agent_dir / name).is_file()]
    candidates.extend(
        path
        for path in sorted(agent_dir.rglob("*"))
        if path.is_file() and path.name not in {"trajectory.json"} and path not in candidates
    )
    chunks: list[str] = []
    total = 0
    for path in candidates:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:4096]:
            continue
        text = data.decode("utf-8", errors="replace")
        remaining = limit_bytes - total
        if remaining <= 0:
            break
        chunks.append(text[:remaining])
        total += len(text[:remaining])
    return "\n".join(chunks)
