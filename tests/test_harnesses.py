from __future__ import annotations

import json

import pytest

from agent_skill_eval.harnesses import (
    ClaudeCodeHarness,
    CodexHarness,
    FakeHarness,
    OpenCodeHarness,
    detect_skill_invocation,
)
from agent_skill_eval.models import AgentType


class TestOpenCodeHarnessParseOutput:
    def test_extracts_text_messages(self, tmp_path):
        harness = OpenCodeHarness(tmp_path)
        stdout = json.dumps({"type": "text", "part": {"text": "Hello world"}})
        output, timing = harness.parse_output(stdout, "")
        assert "Hello world" in output

    def test_extracts_token_usage(self, tmp_path):
        harness = OpenCodeHarness(tmp_path)
        stdout = json.dumps(
            {
                "type": "step_finish",
                "part": {"tokens": {"input": 100, "output": 50, "cache": {"read": 20}}},
            }
        )
        output, timing = harness.parse_output(stdout, "")
        assert timing.input_tokens == 100
        assert timing.output_tokens == 50
        assert timing.cached_tokens == 20
        assert timing.total_tokens == 150

    def test_accumulates_cost_across_steps(self, tmp_path):
        harness = OpenCodeHarness(tmp_path)
        events = [
            json.dumps({"type": "step_finish", "part": {"tokens": {}, "cost": 0.001}}),
            json.dumps({"type": "step_finish", "part": {"tokens": {}, "cost": 0.002}}),
        ]
        output, timing = harness.parse_output("\n".join(events), "")
        assert timing.cost_usd == pytest.approx(0.003)

    def test_handles_multiple_events(self, tmp_path):
        harness = OpenCodeHarness(tmp_path)
        events = [
            json.dumps({"type": "text", "part": {"text": "First"}}),
            json.dumps({"type": "text", "part": {"text": "Second"}}),
        ]
        stdout = "\n".join(events)
        output, timing = harness.parse_output(stdout, "")
        assert "First" in output
        assert "Second" in output

    def test_handles_empty_output(self, tmp_path):
        harness = OpenCodeHarness(tmp_path)
        output, timing = harness.parse_output("", "")
        assert output == ""
        assert timing.total_tokens == 0

    def test_skips_invalid_json(self, tmp_path):
        harness = OpenCodeHarness(tmp_path)
        stdout = "not json\n" + json.dumps({"type": "text", "part": {"text": "valid"}})
        output, timing = harness.parse_output(stdout, "")
        assert "valid" in output

    def test_strips_terminal_escapes_before_json(self, tmp_path):
        # Real opencode runs on some terminals prefix the JSON event with an
        # OSC window-title sequence and ANSI color codes on the same line.
        harness = OpenCodeHarness(tmp_path)
        event = json.dumps({"type": "text", "part": {"text": "Hello after escapes"}})
        line = "\x1b]0;opencode - running task\x07\x1b[32m" + event + "\x1b[0m"
        output, timing = harness.parse_output(line, "")
        assert output == "Hello after escapes"

    def test_strips_ansi_split_across_events(self, tmp_path):
        harness = OpenCodeHarness(tmp_path)
        events = [
            "\x1b]0;title\x1b\\" + json.dumps({"type": "text", "part": {"text": "First"}}),
            "\x1b[1;31m" + json.dumps({"type": "step_finish", "part": {"tokens": {"input": 7, "output": 3}}}),
        ]
        output, timing = harness.parse_output("\n".join(events), "")
        assert output == "First"
        assert timing.input_tokens == 7
        assert timing.output_tokens == 3


class TestOpenCodeHarnessBuildCommand:
    def test_pins_invocation(self, tmp_path, monkeypatch):
        # Pinned V1 invocation; V2 is covered by TestOpenCodeCommandVersions.
        from agent_skill_eval import harnesses

        monkeypatch.setattr(harnesses, "opencode_major_version", lambda: 1)
        harness = OpenCodeHarness(tmp_path)
        cmd = harness.build_command("fix it", tmp_path)
        assert cmd == [
            "opencode",
            "run",
            "--dir",
            str(tmp_path),
            "--format",
            "json",
            "--dangerously-skip-permissions",
            "fix it",
        ]

    def test_model_flag_inserted_before_prompt(self, tmp_path):
        harness = OpenCodeHarness(tmp_path, model="deepseek/deepseek-v4-flash:free")
        cmd = harness.build_command("fix it", tmp_path)
        assert cmd[-3:] == ["--model", "deepseek/deepseek-v4-flash:free", "fix it"]


class TestClaudeCodeHarnessParseOutput:
    def test_extracts_result(self, tmp_path):
        harness = ClaudeCodeHarness(tmp_path)
        stdout = json.dumps({"result": "Task completed", "usage": {"input_tokens": 200, "output_tokens": 100}})
        output, timing = harness.parse_output(stdout, "")
        assert output == "Task completed"
        assert timing.input_tokens == 200
        assert timing.output_tokens == 100

    def test_extracts_cached_tokens(self, tmp_path):
        harness = ClaudeCodeHarness(tmp_path)
        stdout = json.dumps(
            {
                "result": "done",
                "usage": {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 30},
            }
        )
        output, timing = harness.parse_output(stdout, "")
        assert timing.cached_tokens == 30

    def test_extracts_cost(self, tmp_path):
        harness = ClaudeCodeHarness(tmp_path)
        stdout = json.dumps({"result": "done", "usage": {}, "total_cost_usd": 0.0123})
        output, timing = harness.parse_output(stdout, "")
        assert timing.cost_usd == 0.0123

    def test_extracts_cache_creation_tokens(self, tmp_path):
        harness = ClaudeCodeHarness(tmp_path)
        stdout = json.dumps(
            {
                "result": "done",
                "usage": {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 40},
            }
        )
        output, timing = harness.parse_output(stdout, "")
        assert timing.cache_creation_tokens == 40

    def test_handles_invalid_json(self, tmp_path):
        harness = ClaudeCodeHarness(tmp_path)
        output, timing = harness.parse_output("plain text output", "")
        assert output == "plain text output"


class TestClaudeCodeHarnessBuildCommand:
    def test_pins_invocation(self, tmp_path):
        harness = ClaudeCodeHarness(tmp_path)
        cmd = harness.build_command("fix it", tmp_path)
        assert cmd == [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
            "fix it",
        ]

    def test_model_flag_inserted_before_prompt(self, tmp_path):
        harness = ClaudeCodeHarness(tmp_path, model="claude-haiku-4-5-20251001")
        cmd = harness.build_command("fix it", tmp_path)
        assert cmd[-3:] == ["--model", "claude-haiku-4-5-20251001", "fix it"]


class TestCodexHarnessParseOutput:
    def test_extracts_messages(self, tmp_path):
        harness = CodexHarness(tmp_path)
        stdout = json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "Codex response"},
            }
        )
        output, timing = harness.parse_output(stdout, "")
        assert "Codex response" in output

    def test_extracts_usage(self, tmp_path):
        harness = CodexHarness(tmp_path)
        stdout = json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 300, "output_tokens": 150, "cached_input_tokens": 50},
            }
        )
        output, timing = harness.parse_output(stdout, "")
        assert timing.input_tokens == 300
        assert timing.output_tokens == 150
        assert timing.cached_tokens == 50

    def test_handles_empty_output(self, tmp_path):
        harness = CodexHarness(tmp_path)
        output, timing = harness.parse_output("", "")
        assert output == ""
        assert timing.total_tokens == 0

    def test_build_command_skips_git_trust_check(self, tmp_path):
        harness = CodexHarness(tmp_path)
        cmd = harness.build_command("fix it", tmp_path)
        assert "--skip-git-repo-check" in cmd
        assert "--full-auto" not in cmd

    def test_pins_invocation(self, tmp_path):
        harness = CodexHarness(tmp_path, model="gpt-5.4-mini")
        cmd = harness.build_command("fix it", tmp_path)
        assert cmd == [
            "codex",
            "exec",
            "--json",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "--model",
            "gpt-5.4-mini",
            "fix it",
        ]


class TestFakeHarness:
    def test_run_formats_when_fake_skill_is_installed(self, tmp_path):
        skill_dir = tmp_path / ".fake" / "skills" / "format-json"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Format JSON")

        harness = FakeHarness(tmp_path)
        output, timing, stdout, stderr = harness.run("Format JSON", tmp_path / "outputs")

        assert "formatted-json-ok" in output
        assert stdout == output
        assert stderr == ""
        assert timing.total_tokens == 1

    def test_run_baseline_does_not_format_without_skill(self, tmp_path):
        harness = FakeHarness(tmp_path)
        output, timing, stdout, stderr = harness.run("Format JSON", tmp_path / "outputs")

        assert "formatted-json-ok" not in output
        assert stdout == output
        assert stderr == ""
        assert timing.total_tokens == 1


class TestDetectSkillInvocation:
    def _opencode_path(self, tmp_path):
        return tmp_path / ".opencode" / "skills" / "demo"

    def test_opencode_skill_tool_invocation(self, tmp_path):
        stdout = json.dumps({"type": "tool_use", "part": {"tool": "skill", "state": {"input": {"name": "demo"}}}})
        invoked, evidence = detect_skill_invocation(
            AgentType.OPENCODE, stdout, "", "demo", [self._opencode_path(tmp_path)]
        )
        assert invoked is True
        assert "skill" in evidence

    def test_opencode_failed_skill_call_is_not_an_invocation(self, tmp_path):
        # The no-skill arm can guess a skill name; the call then fails.
        stdout = json.dumps(
            {
                "type": "tool_use",
                "part": {
                    "tool": "skill",
                    "state": {"status": "error", "input": {"id": "demo"}, "error": "Unable to load skill demo"},
                },
            }
        )
        invoked, evidence = detect_skill_invocation(
            AgentType.OPENCODE, stdout, "", "demo", [self._opencode_path(tmp_path)]
        )
        assert invoked is False
        assert "none invoked" in evidence

    def test_opencode_completed_skill_call_counts(self, tmp_path):
        stdout = json.dumps(
            {
                "type": "tool_use",
                "part": {"tool": "skill", "state": {"status": "completed", "input": {"id": "demo"}}},
            }
        )
        invoked, _ = detect_skill_invocation(AgentType.OPENCODE, stdout, "", "demo", [self._opencode_path(tmp_path)])
        assert invoked is True

    def test_opencode_read_of_skill_md(self, tmp_path):
        stdout = json.dumps(
            {
                "type": "tool_use",
                "part": {
                    "tool": "read",
                    "state": {"input": {"filePath": ".opencode/skills/demo/SKILL.md"}},
                },
            }
        )
        invoked, evidence = detect_skill_invocation(
            AgentType.OPENCODE, stdout, "", "demo", [self._opencode_path(tmp_path)]
        )
        assert invoked is True
        assert "SKILL.md" in evidence

    def test_opencode_tool_events_without_skill_are_false(self, tmp_path):
        stdout = json.dumps({"type": "tool_use", "part": {"tool": "bash", "state": {"input": {"command": "ls"}}}})
        invoked, evidence = detect_skill_invocation(
            AgentType.OPENCODE, stdout, "", "demo", [self._opencode_path(tmp_path)]
        )
        assert invoked is False
        assert "none invoked" in evidence

    def test_opencode_without_tool_events_is_unknown(self, tmp_path):
        stdout = json.dumps({"type": "text", "part": {"text": "done"}})
        invoked, _ = detect_skill_invocation(AgentType.OPENCODE, stdout, "", "demo", [self._opencode_path(tmp_path)])
        assert invoked is None

    def test_codex_command_read_of_skill_md(self, tmp_path):
        stdout = json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "cat .agents/skills/demo/SKILL.md",
                },
            }
        )
        invoked, evidence = detect_skill_invocation(
            AgentType.CODEX, stdout, "", "demo", [tmp_path / ".agents" / "skills" / "demo"]
        )
        assert invoked is True
        assert "SKILL.md" in evidence

    def test_codex_without_skill_event_is_unknown(self, tmp_path):
        stdout = json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "command": "pytest -q"},
            }
        )
        invoked, _ = detect_skill_invocation(
            AgentType.CODEX, stdout, "", "demo", [tmp_path / ".agents" / "skills" / "demo"]
        )
        assert invoked is None

    def test_claude_code_path_in_logs_is_true(self, tmp_path):
        invoked, _ = detect_skill_invocation(
            AgentType.CLAUDE_CODE,
            "",
            "Read .claude/skills/demo/SKILL.md",
            "demo",
            [tmp_path / ".claude" / "skills" / "demo"],
        )
        assert invoked is True

    def test_claude_code_without_path_is_unknown(self, tmp_path):
        invoked, _ = detect_skill_invocation(
            AgentType.CLAUDE_CODE,
            json.dumps({"result": "done"}),
            "",
            "demo",
            [tmp_path / ".claude" / "skills" / "demo"],
        )
        assert invoked is None

    def test_fake_harness_reports_installed_skill_use(self, tmp_path):
        invoked, _ = detect_skill_invocation(AgentType.FAKE, "", "", "demo", [tmp_path / ".fake" / "skills" / "demo"])
        assert invoked is True
        invoked, _ = detect_skill_invocation(AgentType.FAKE, "", "", "demo", [])
        assert invoked is False


class TestOpenCodeCommandVersions:
    """OpenCode V1 and V2 take different flags for the same run."""

    @pytest.fixture(autouse=True)
    def _clear_version_cache(self):
        from agent_skill_eval import harnesses

        harnesses.opencode_major_version.cache_clear()
        yield
        harnesses.opencode_major_version.cache_clear()

    def test_v1_uses_dir_and_skip_permissions(self, tmp_path, monkeypatch):
        from agent_skill_eval import harnesses

        monkeypatch.setattr(harnesses, "get_cli_version", lambda agent: "1.19.5")
        cmd = OpenCodeHarness(tmp_path, model="opencode-go/m").build_command("hi", tmp_path)
        assert cmd[:2] == ["opencode", "run"]
        assert "--dir" in cmd and str(tmp_path) in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert cmd[-1] == "hi"

    def test_v2_uses_auto_standalone_and_no_dir(self, tmp_path, monkeypatch):
        from agent_skill_eval import harnesses

        monkeypatch.setattr(harnesses, "get_cli_version", lambda agent: "opencode v2.2.6")
        cmd = OpenCodeHarness(tmp_path, model="opencode-go/m").build_command("hi", tmp_path)
        assert "--dir" not in cmd
        assert "--dangerously-skip-permissions" not in cmd
        assert "--auto" in cmd and "--standalone" in cmd
        assert cmd[cmd.index("--format") + 1] == "json"
        assert cmd[cmd.index("--model") + 1] == "opencode-go/m"
        assert cmd[-1] == "hi"

    def test_unknown_version_falls_back_to_v1(self, tmp_path, monkeypatch):
        from agent_skill_eval import harnesses

        monkeypatch.setattr(harnesses, "get_cli_version", lambda agent: None)
        assert harnesses.opencode_major_version() == 1
