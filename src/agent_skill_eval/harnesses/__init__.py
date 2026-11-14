from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from agent_skill_eval.models import AgentType, TimingData

DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_MAX_RETRIES = 1
RETRY_BACKOFF_SECONDS = 2.0

# Terminal escape sequences an agent CLI may print on the same line as a JSON
# event: OSC (window title, e.g. ``\x1b]0;title\x07``), CSI (colors/cursor),
# and the two-character escapes. Codex/OpenCode emit these when stdout is a
# TTY-like pipe on some terminals, which used to make json.loads drop the line.
_TERMINAL_ESCAPE_RE = re.compile(
    r"""
    \x1b
    (?:
        \[[0-?]*[ -/]*[@-~]        # CSI ... final byte
      | \][^\x07\x1b]*(?:\x07|\x1b\\)  # OSC ... BEL or ST
      | [PX^_][^\x1b]*(?:\x1b\\)   # DCS/SOS/PM/APC ... ST
      | [@-_]                      # Fe two-character escape
    )
    """,
    re.VERBOSE,
)


def strip_terminal_escapes(text: str) -> str:
    """Remove ANSI CSI/OSC escape sequences from a log line."""
    return _TERMINAL_ESCAPE_RE.sub("", text)


def _iter_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _iter_strings(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _iter_strings(item)]
    return []


def _skill_markers(skill_name: str, installed_paths: list[Path]) -> tuple[str, ...]:
    markers = {f"{skill_name}/SKILL.md"}
    for path in installed_paths:
        markers.add(path.as_posix())
        # Logs may print the path relative to the workspace.
        parts = path.parts
        if "skills" in parts:
            index = parts.index("skills")
            if index > 0 and index + 1 < len(parts):
                markers.add("/".join(parts[index - 1 :]))
    return tuple(marker.lower() for marker in markers)


def _mentions_skill(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower().replace("\\", "/")
    return any(marker in lowered for marker in markers)


def detect_skill_invocation(
    agent_type: AgentType,
    stdout: str,
    stderr: str,
    skill_name: str,
    installed_paths: list[Path],
) -> tuple[Optional[bool], str]:
    """Whether the agent actually loaded/invoked the installed skill.

    Returns ``(True, evidence)`` when a read or skill tool call for the
    installed SKILL.md is visible, ``(False, evidence)`` only when the log
    format proves no invocation happened, and ``(None, evidence)`` when the
    agent exposes too little telemetry (unknown must not count as false).
    """
    markers = _skill_markers(skill_name, installed_paths)

    if agent_type == AgentType.FAKE:
        if installed_paths:
            return True, "fake harness simulates skill use when the skill is installed"
        return False, "fake harness has no installed skill"

    if agent_type == AgentType.OPENCODE:
        saw_tool_event = False
        for line in stdout.splitlines():
            try:
                event = json.loads(strip_terminal_escapes(line))
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            part = event.get("part") if isinstance(event.get("part"), dict) else {}
            if event.get("type") not in {"tool_use", "tool"} and part.get("type") not in {"tool", "tool_use"}:
                continue
            saw_tool_event = True
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            if str(state.get("status", "")).lower() in {"error", "failed"} or state.get("error"):
                # A failed call (for example "Unable to load skill pdf" in the
                # no-skill arm) did not load anything.
                continue
            tool = str(part.get("tool") or part.get("tool_name") or part.get("name") or event.get("tool") or "")
            tool_input = part.get("input", event.get("input"))
            if tool_input is None and isinstance(part.get("state"), dict):
                tool_input = part["state"].get("input")
            strings = _iter_strings(tool_input)
            if tool.lower() == "skill":
                if not strings or any(skill_name.lower() in text.lower() for text in strings):
                    return True, f"opencode tool 'skill' invoked {skill_name}"
            if any(_mentions_skill(text, markers) for text in strings):
                return True, "opencode read the installed SKILL.md"
        if saw_tool_event:
            return False, "opencode tool events present, but none invoked the skill"
        return None, "opencode log has no structured tool events"

    if agent_type == AgentType.CODEX:
        for line in stdout.splitlines():
            try:
                event = json.loads(strip_terminal_escapes(line))
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            for obj in (event, item):
                if obj.get("type") == "command_execution":
                    command = str(obj.get("command") or "")
                    if _mentions_skill(command, markers):
                        return True, "codex command read the installed SKILL.md"
                if obj.get("type") in {"tool_call", "mcp_tool_call", "function_call"}:
                    name = str(obj.get("tool_name") or obj.get("name") or "")
                    arguments = obj.get("arguments") or obj.get("input") or {}
                    if name.lower() == "skill" and (
                        not arguments or any(skill_name.lower() in text.lower() for text in _iter_strings(arguments))
                    ):
                        return True, f"codex tool 'skill' invoked {skill_name}"
                if obj.get("type") in {"file_read", "read_file"}:
                    if any(_mentions_skill(text, markers) for text in _iter_strings(obj)):
                        return True, "codex read the installed SKILL.md"
        return None, "codex log has no structured skill-read event"

    if agent_type == AgentType.CLAUDE_CODE:
        if _mentions_skill(stdout + "\n" + stderr, markers):
            return True, "installed SKILL.md path appears in claude-code logs"
        return None, "claude-code JSON output exposes no tool-call events"

    return None, "skill invocation detection is unavailable for this harness"


def _rate(entry: dict, field: str) -> float:
    try:
        return float(entry.get(field) or 0.0)
    except (TypeError, ValueError):
        return 0.0


class AgentHarness(ABC):
    agent_type: AgentType
    # Environment variable each agent CLI reads its API base URL from.
    base_url_env: Optional[str] = None
    # Command that prints the agent CLI's version (None = no probe).
    version_command: Optional[list[str]] = None
    # Whether build_command can pass --reasoning-effort through to the CLI.
    supports_reasoning_effort: bool = False

    def __init__(
        self,
        workspace: Path,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        pricing: Optional[dict[str, dict]] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.workspace = workspace
        self.model = model
        self.base_url = base_url
        # Only meaningful when supports_reasoning_effort; callers should
        # treat the effective value as None for harnesses that ignore it.
        self.reasoning_effort = reasoning_effort if self.supports_reasoning_effort else None
        self.timeout = timeout or int(os.environ.get("ASE_AGENT_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
        if max_retries is None:
            max_retries = int(os.environ.get("ASE_AGENT_RETRIES", DEFAULT_MAX_RETRIES))
        self.max_retries = max_retries
        # User-provided model -> rate map (--pricing-config), used to compute
        # cost_usd when the CLI itself reports nothing.
        self.pricing = pricing or {}

    @abstractmethod
    def build_command(self, prompt: str, output_dir: Path) -> list[str]:
        pass

    @abstractmethod
    def parse_output(self, stdout: str, stderr: str) -> tuple[str, TimingData]:
        pass

    def non_cached_input_tokens(self, timing: TimingData) -> int:
        """Input tokens billed at the non-cached rate.

        Default for CLIs that report input_tokens INCLUDING cache reads
        (codex); claude-code and opencode report them separately and
        override this.
        """
        return max(timing.input_tokens - timing.cached_tokens, 0)

    def _cost_from_pricing_config(self, timing: TimingData) -> Optional[float]:
        """Compute cost from a user-provided --pricing-config, or None.

        Rates are USD per token (same semantics as OpenRouter's pricing
        fields): prompt, completion, input_cache_read, input_cache_write.
        Matched by exact model string.
        """
        if not self.model or self.model not in self.pricing:
            return None
        if timing.total_tokens <= 0 and timing.cached_tokens <= 0 and timing.cache_creation_tokens <= 0:
            return None
        entry = self.pricing[self.model]
        prompt = _rate(entry, "prompt")
        completion = _rate(entry, "completion")
        if prompt <= 0 and completion <= 0:
            return None
        cache_read = _rate(entry, "input_cache_read") or prompt
        return (
            self.non_cached_input_tokens(timing) * prompt
            + timing.cached_tokens * cache_read
            + timing.output_tokens * completion
            + timing.cache_creation_tokens * _rate(entry, "input_cache_write")
        )

    def finalize_timing(self, timing: TimingData) -> None:
        """Post-process timing after a run (e.g. cost reconciliation)."""
        if timing.cost_usd is not None and timing.cost_usd_source is None:
            timing.cost_usd_source = "cli"
            return
        if timing.cost_usd is None:
            computed = self._cost_from_pricing_config(timing)
            if computed is not None:
                timing.cost_usd = computed
                timing.cost_usd_source = "pricing-config"

    def _build_env(self) -> Optional[dict[str, str]]:
        if self.base_url and self.base_url_env:
            env = dict(os.environ)
            env[self.base_url_env] = self.base_url
            return env
        return None

    def _workspace_fingerprint(self) -> Optional[tuple[str, str]]:
        """(HEAD sha, porcelain status) of the workspace, or None outside git.

        Used to refuse retries after a failed attempt that already mutated
        the workspace — re-running there would grade the union of both
        attempts' side effects.
        """
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=15,
                check=False,
            )
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=15,
                check=False,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        if head.returncode != 0 or status.returncode != 0:
            return None
        return (head.stdout.strip(), status.stdout)

    def run(self, prompt: str, output_dir: Path) -> tuple[str, TimingData, str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = self.build_command(prompt, output_dir)
        env = self._build_env()

        attempts = self.max_retries + 1
        stdout = ""
        stderr = ""
        exit_code: Optional[int] = None
        timed_out = False
        retries_used = 0
        start = time.time()
        fingerprint_before = self._workspace_fingerprint()

        for attempt in range(attempts):
            if attempt > 0:
                retries_used = attempt
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.workspace,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    env=env,
                )
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                exit_code = result.returncode
                timed_out = False
                if result.returncode == 0:
                    break
            except subprocess.TimeoutExpired as e:
                stdout = (
                    (e.stdout or b"").decode("utf-8", errors="replace")
                    if isinstance(e.stdout, bytes)
                    else (e.stdout or "")
                )
                stderr = (
                    (e.stderr or b"").decode("utf-8", errors="replace")
                    if isinstance(e.stderr, bytes)
                    else (e.stderr or "")
                )
                stderr += f"\n[agent-skill-eval] timed out after {self.timeout}s (attempt {attempt + 1}/{attempts})"
                exit_code = None
                timed_out = True

            # Only retry into a pristine workspace: if the failed attempt
            # already committed/wrote files, a second run would produce a
            # graded union of both attempts.
            if attempt < attempts - 1 and fingerprint_before is not None:
                if self._workspace_fingerprint() != fingerprint_before:
                    stderr += "\n[agent-skill-eval] not retrying: the failed attempt modified the workspace"
                    break

        duration_ms = int((time.time() - start) * 1000)

        final_output, timing = self.parse_output(stdout, stderr)
        if timing.duration_ms == 0:
            timing.duration_ms = duration_ms
        timing.exit_code = exit_code
        timing.timed_out = timed_out
        timing.retries = retries_used
        # Persist the billable input split using this harness's semantics;
        # downstream readers must not re-derive it (codex/opencode include
        # cache reads in input_tokens, claude-code does not).
        timing.non_cached_input_tokens = self.non_cached_input_tokens(timing)
        self.finalize_timing(timing)

        with open(output_dir / "stdout.log", "w", encoding="utf-8") as f:
            f.write(stdout)
        with open(output_dir / "stderr.log", "w", encoding="utf-8") as f:
            f.write(stderr)

        return final_output, timing, stdout, stderr


class OpenCodeHarness(AgentHarness):
    agent_type = AgentType.OPENCODE
    base_url_env = "OPENAI_BASE_URL"
    version_command = ["opencode", "--version"]

    def build_command(self, prompt: str, output_dir: Path) -> list[str]:
        if opencode_major_version() >= 2:
            # OpenCode V2 has no --dir (the process cwd is already the
            # workspace) and no --dangerously-skip-permissions (--auto
            # approves everything not explicitly denied). --standalone gives
            # each run a private server instead of the shared background
            # service, so parallel runs do not share state.
            cmd = ["opencode", "run", "--standalone", "--format", "json", "--auto"]
        else:
            cmd = [
                "opencode",
                "run",
                "--dir",
                str(self.workspace),
                "--format",
                "json",
                "--dangerously-skip-permissions",
            ]
        if self.model:
            cmd.extend(["--model", self.model])
        cmd.append(prompt)
        return cmd

    def non_cached_input_tokens(self, timing: TimingData) -> int:
        # OpenCode reports input tokens EXCLUDING cache reads (like
        # claude-code): live runs show cached consistently larger than input,
        # which is impossible if input contained the cache reads. The base
        # class subtraction would clamp this to a bogus 0.
        return timing.input_tokens

    def parse_output(self, stdout: str, stderr: str) -> tuple[str, TimingData]:
        timing = TimingData()
        messages = []
        # None until at least one step reports a cost field: a run whose
        # steps never report cost is "cost unavailable", not a free run.
        cost: Optional[float] = None

        for line in stdout.strip().split("\n"):
            # A line may start with OSC/ANSI escapes printed by the terminal
            # before the JSON event; strip them so json.loads sees the event.
            line = strip_terminal_escapes(line)
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                etype = event.get("type", "")

                if etype == "text":
                    part = event.get("part", {})
                    text = part.get("text", "")
                    if text:
                        messages.append(text)

                elif etype == "step_finish":
                    part = event.get("part", {})
                    tokens = part.get("tokens", {})
                    timing.input_tokens += tokens.get("input", 0)
                    timing.output_tokens += tokens.get("output", 0)
                    cache = tokens.get("cache", {})
                    timing.cached_tokens += cache.get("read", 0)
                    # Only set when a step actually reports the field: a run
                    # with no reasoning telemetry stays None (unknown), which
                    # is not the same as reasoning=0.
                    if "reasoning" in tokens:
                        timing.reasoning_output_tokens = (timing.reasoning_output_tokens or 0) + (
                            tokens.get("reasoning") or 0
                        )
                    if "cost" in part:
                        cost = (cost or 0.0) + (part.get("cost") or 0)

            except json.JSONDecodeError:
                continue

        timing.cost_usd = cost
        timing.total_tokens = timing.input_tokens + timing.output_tokens
        return "\n".join(messages), timing


class ClaudeCodeHarness(AgentHarness):
    agent_type = AgentType.CLAUDE_CODE
    base_url_env = "ANTHROPIC_BASE_URL"
    version_command = ["claude", "--version"]

    def build_command(self, prompt: str, output_dir: Path) -> list[str]:
        cmd = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        cmd.append(prompt)
        return cmd

    def parse_output(self, stdout: str, stderr: str) -> tuple[str, TimingData]:
        timing = TimingData()
        final_text = ""

        try:
            data = json.loads(stdout)
            if isinstance(data, dict):
                final_text = data.get("result", "")
                usage = data.get("usage", {})
                timing.input_tokens = usage.get("input_tokens", 0)
                timing.output_tokens = usage.get("output_tokens", 0)
                timing.cached_tokens = usage.get("cache_read_input_tokens", 0)
                timing.cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
                raw_cost = data.get("total_cost_usd")
                timing.cost_usd = float(raw_cost) if raw_cost is not None else None
                timing.total_tokens = timing.input_tokens + timing.output_tokens
                duration_s = data.get("duration_ms", 0)
                if duration_s:
                    timing.duration_ms = duration_s
        except json.JSONDecodeError:
            final_text = stdout

        return final_text, timing

    def finalize_timing(self, timing: TimingData) -> None:
        """Reconcile the CLI's cost estimate when running via OpenRouter.

        The claude CLI prices runs at Anthropic list prices regardless of the
        endpoint it talks to. When ANTHROPIC_BASE_URL points at OpenRouter,
        actual billing follows OpenRouter's per-model pricing, so we recompute
        from token counts (the generation API would be exact but needs
        per-request ids the CLI does not expose). The CLI's own number is kept
        in cost_usd_cli.
        """
        base = self.base_url or os.environ.get("ANTHROPIC_BASE_URL", "")
        if "openrouter" not in base.lower():
            super().finalize_timing(timing)
            return

        from agent_skill_eval.openrouter import reconcile_claude_cost

        reconciled = reconcile_claude_cost(timing, self.model)
        if reconciled is None:
            if timing.cost_usd is None:
                # The CLI reported no cost either: fall back to the generic
                # path (pricing config, or genuinely unavailable).
                super().finalize_timing(timing)
            else:
                timing.cost_usd_source = "cli-unreconciled"
            return
        timing.cost_usd_cli = timing.cost_usd
        timing.cost_usd = reconciled
        timing.cost_usd_source = "openrouter-pricing"

    def non_cached_input_tokens(self, timing: TimingData) -> int:
        # Anthropic usage reports input_tokens EXCLUDING cache reads.
        return timing.input_tokens


class CodexHarness(AgentHarness):
    agent_type = AgentType.CODEX
    base_url_env = "OPENAI_BASE_URL"
    version_command = ["codex", "--version"]
    supports_reasoning_effort = True

    def build_command(self, prompt: str, output_dir: Path) -> list[str]:
        cmd = [
            "codex",
            "exec",
            "--json",
            "--sandbox",
            "workspace-write",
            # Eval workspaces are freshly git-inited dirs codex has never
            # seen; without this codex blocks on its trust prompt.
            "--skip-git-repo-check",
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        if self.reasoning_effort:
            # Without this, codex silently inherits model_reasoning_effort
            # from ~/.codex/config.toml — two users running the "same" eval
            # can benchmark different reasoning settings.
            cmd.extend(["-c", f'model_reasoning_effort="{self.reasoning_effort}"'])
        cmd.append(prompt)
        return cmd

    def parse_output(self, stdout: str, stderr: str) -> tuple[str, TimingData]:
        timing = TimingData()
        messages = []

        for line in stdout.strip().split("\n"):
            line = strip_terminal_escapes(line)
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                etype = event.get("type", "")

                if etype == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        messages.append(item.get("text", ""))

                elif etype == "turn.completed":
                    usage = event.get("usage", {})
                    timing.input_tokens += usage.get("input_tokens", 0)
                    timing.output_tokens += usage.get("output_tokens", 0)
                    timing.cached_tokens += usage.get("cached_input_tokens", 0)
                    # Only set when the turn reports the field; absent
                    # telemetry stays None (unknown), not 0.
                    if "reasoning_output_tokens" in usage:
                        timing.reasoning_output_tokens = (timing.reasoning_output_tokens or 0) + (
                            usage.get("reasoning_output_tokens") or 0
                        )

            except json.JSONDecodeError:
                continue

        timing.total_tokens = timing.input_tokens + timing.output_tokens
        return "\n".join(messages), timing


class FakeHarness(AgentHarness):
    agent_type = AgentType.FAKE

    def build_command(self, prompt: str, output_dir: Path) -> list[str]:
        return ["fake-agent", prompt]

    def parse_output(self, stdout: str, stderr: str) -> tuple[str, TimingData]:
        # The fake agent makes no API calls, so its cost is a real 0.0
        # (known-free), not an unavailable None.
        timing = TimingData(total_tokens=1, input_tokens=1, duration_ms=1, cost_usd=0.0)
        return stdout, timing

    def run(self, prompt: str, output_dir: Path) -> tuple[str, TimingData, str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        skill_installed = any((self.workspace / ".fake" / "skills").glob("*/SKILL.md"))
        for marker in ("fake-create-file:", "fake_create_file:"):
            if marker in prompt:
                rel = prompt.split(marker, 1)[1].split()[0].strip().strip("`'\"")
                target = self.workspace / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("created by fake harness\n", encoding="utf-8")
        if skill_installed and ("format" in prompt.lower() or "$format-json" in prompt.lower()):
            final_output = '{\n  "formatted": true,\n  "status": "formatted-json-ok"\n}'
        else:
            final_output = "fake-agent baseline output: no formatting performed"

        stdout = final_output
        stderr = ""
        timing = TimingData(
            total_tokens=1, input_tokens=1, output_tokens=0, cached_tokens=0, duration_ms=1, exit_code=0, cost_usd=0.0
        )
        timing.non_cached_input_tokens = self.non_cached_input_tokens(timing)

        with open(output_dir / "stdout.log", "w", encoding="utf-8") as f:
            f.write(stdout)
        with open(output_dir / "stderr.log", "w", encoding="utf-8") as f:
            f.write(stderr)

        return final_output, timing, stdout, stderr


HARNESSES: dict[AgentType, type[AgentHarness]] = {
    AgentType.OPENCODE: OpenCodeHarness,
    AgentType.CLAUDE_CODE: ClaudeCodeHarness,
    AgentType.CODEX: CodexHarness,
    AgentType.FAKE: FakeHarness,
}


def get_harness(
    agent_type: AgentType,
    workspace: Path,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    pricing: Optional[dict[str, dict]] = None,
    reasoning_effort: Optional[str] = None,
) -> AgentHarness:
    cls = HARNESSES[agent_type]
    return cls(
        workspace,
        model,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        pricing=pricing,
        reasoning_effort=reasoning_effort,
    )


@functools.lru_cache(maxsize=1)
def opencode_major_version() -> int:
    """Major version of the installed OpenCode CLI; 1 when unknown.

    ``opencode --version`` prints ``1.x.y`` (V1) or ``opencode v2.x.y`` (V2).
    """
    version = get_cli_version(AgentType.OPENCODE) or ""
    match = re.search(r"(\d+)\.\d+", version)
    return int(match.group(1)) if match else 1


def get_cli_version(agent_type: AgentType) -> Optional[str]:
    """Best-effort agent CLI version (first line of ``<cli> --version``).

    Recorded in run_meta.json so a result can be tied to the exact agent
    binary that produced it. None when the agent has no version command or
    the probe fails — never raises.
    """
    command = HARNESSES[agent_type].version_command
    if not command:
        return None
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, check=False
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    first_line = (result.stdout or "").strip().splitlines()
    return first_line[0].strip() if first_line else None
