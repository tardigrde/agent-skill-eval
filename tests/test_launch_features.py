"""Tests for the launch feature set: per-agent deltas, pass@k, skipped grading,
agent model mapping, harness resilience, and the validate/list/compare CLI."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from agent_skill_eval.cli import _discover_eval_suites, _discover_skills, _grader_settings, _parse_agent_models, app
from agent_skill_eval.graders import (
    DEFAULT_GRADER_MODEL,
    OPENWEBUI_GRADER_BASE_URL,
    OPENWEBUI_GRADER_MODEL,
    grade_assertions,
)
from agent_skill_eval.harnesses import ClaudeCodeHarness, CodexHarness, OpenCodeHarness, get_harness
from agent_skill_eval.models import AgentType, EvalSuite
from agent_skill_eval.runner import compute_benchmark, compute_stats

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent


def _result(agent: str, with_skill: bool, pass_rate: float, eval_id: str = "e1", total: int = 2) -> dict:
    return {
        "eval_id": eval_id,
        "agent": agent,
        "with_skill": with_skill,
        "grading": {"summary": {"pass_rate": pass_rate, "total": total, "passed": int(pass_rate * total)}},
        "timing": {"duration_ms": 1000, "total_tokens": 100},
    }


class TestMultiAgentDelta:
    def test_per_agent_deltas_computed(self):
        results = {
            "a": _result("opencode", True, 1.0),
            "b": _result("opencode", False, 0.5),
            "c": _result("claude-code", True, 0.5),
            "d": _result("claude-code", False, 0.5),
        }
        res = compute_benchmark(results, [AgentType.OPENCODE, AgentType.CLAUDE_CODE], with_baseline=True)
        assert res.deltas["opencode"].pass_rate == pytest.approx(0.5)
        assert res.deltas["claude-code"].pass_rate == pytest.approx(0.0)
        # Backward-compat field is None for multi-agent runs, never a silent 0.
        assert res.delta is None

    def test_single_agent_delta_still_populated(self):
        results = {
            "a": _result("opencode", True, 1.0),
            "b": _result("opencode", False, 0.0),
        }
        res = compute_benchmark(results, [AgentType.OPENCODE], with_baseline=True)
        assert res.delta is not None
        assert res.delta.pass_rate == pytest.approx(1.0)
        assert res.deltas["opencode"].pass_rate == pytest.approx(1.0)

    def test_no_baseline_no_deltas(self):
        results = {"a": _result("opencode", True, 1.0)}
        res = compute_benchmark(results, [AgentType.OPENCODE], with_baseline=False)
        assert res.deltas == {}
        assert res.delta is None


class TestPassAtK:
    def test_pass_at_k_any_run_counts(self):
        # eval e1: 1 of 3 runs fully passed; eval e2: 0 of 3.
        results = [
            _result("opencode", True, 1.0, eval_id="e1"),
            _result("opencode", True, 0.5, eval_id="e1"),
            _result("opencode", True, 0.0, eval_id="e1"),
            _result("opencode", True, 0.5, eval_id="e2"),
            _result("opencode", True, 0.5, eval_id="e2"),
            _result("opencode", True, 0.0, eval_id="e2"),
        ]
        stats = compute_stats(results)
        assert stats.k == 3
        assert stats.pass_at_k == pytest.approx(0.5)
        assert stats.full_pass_rate == pytest.approx(1 / 6)

    def test_single_run_pass_at_k_equals_full_pass_rate(self):
        results = [
            _result("opencode", True, 1.0, eval_id="e1"),
            _result("opencode", True, 0.0, eval_id="e2"),
        ]
        stats = compute_stats(results)
        assert stats.k == 1
        assert stats.pass_at_k == pytest.approx(stats.full_pass_rate) == pytest.approx(0.5)

    def test_pass_hat_k_requires_every_attempt(self):
        results = [
            _result("opencode", True, 1.0, eval_id="e1"),
            _result("opencode", True, 1.0, eval_id="e1"),
            _result("opencode", True, 1.0, eval_id="e2"),
            _result("opencode", True, 0.0, eval_id="e2"),
        ]
        stats = compute_stats(results)
        assert stats.k == 2
        assert stats.pass_at_k == pytest.approx(1.0)
        assert stats.pass_hat_k == pytest.approx(0.5)
        assert stats.pass_hat_k_wilson_low < 0.5 < stats.pass_hat_k_wilson_high
        assert stats.cases == 2

    def test_percentiles_are_recorded(self):
        results = [
            _result("opencode", True, 1.0, eval_id="e1"),
            _result("opencode", True, 1.0, eval_id="e2"),
            _result("opencode", True, 1.0, eval_id="e3"),
        ]
        stats = compute_stats(results)
        assert stats.tokens.median == 100.0
        assert stats.tokens.p90 == 100.0
        assert stats.time_seconds.median == 1.0
        assert stats.pass_rate.n == 3


class TestSkippedGrading:
    def test_unknown_assertion_without_llm_grader_is_skipped_not_failed(self, tmp_path):
        grading = grade_assertions(
            assertions=["The summary is written in a friendly tone"],
            agent_output="hello",
            output_dir=tmp_path,
            expected_output="ok",
            llm_grader=None,
        )
        result = grading.assertion_results[0]
        assert result.skipped is True
        assert result.method == "skipped"
        assert grading.summary.skipped == 1
        assert grading.summary.failed == 0
        assert grading.summary.pass_rate == 0.0

    def test_skipped_assertions_do_not_drag_down_pass_rate(self, tmp_path):
        grading = grade_assertions(
            assertions=[
                'The output contains "hello"',
                "The summary is written in a friendly tone",
            ],
            agent_output="hello world",
            output_dir=tmp_path,
            expected_output="ok",
            llm_grader=None,
        )
        assert grading.summary.passed == 1
        assert grading.summary.skipped == 1
        assert grading.summary.pass_rate == pytest.approx(1.0)

    def test_llm_error_marks_assertions_skipped(self, tmp_path, monkeypatch):
        from agent_skill_eval.graders import LLMGrader

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        grader = LLMGrader(model="some/model")
        results = grader.grade(["A subjective assertion"], "output", tmp_path, "expected")
        assert results[0].skipped is True
        assert results[0].method == "llm"
        assert "LLM grading error" in results[0].evidence


class TestOpenWebUIGraderOptions:
    def test_openwebui_rewrites_default_model_and_key_file(self):
        settings = _grader_settings(
            grader_model=DEFAULT_GRADER_MODEL,
            grader_base_url=None,
            grader_key_file=None,
            grader_openwebui=True,
        )

        assert settings["model"] == OPENWEBUI_GRADER_MODEL
        assert settings["base_url"] == OPENWEBUI_GRADER_BASE_URL
        assert settings["key_file"] == Path("/tmp/openwebui.key")
        assert settings["allowed_models"] == (OPENWEBUI_GRADER_MODEL,)
        assert settings["openwebui"] is True

    def test_openwebui_rejects_other_models(self):
        with pytest.raises(Exception):
            _grader_settings(
                grader_model="other/model",
                grader_base_url=None,
                grader_key_file=None,
                grader_openwebui=True,
            )


class TestParseAgentModels:
    def test_agent_scoped_and_bare_specs(self):
        mapping = _parse_agent_models(
            ["claude-code=haiku", "codex=gpt-5-mini"],
            [AgentType.CLAUDE_CODE, AgentType.CODEX],
        )
        assert mapping == {AgentType.CLAUDE_CODE: "haiku", AgentType.CODEX: "gpt-5-mini"}

    def test_bare_spec_applies_to_all_agents(self):
        mapping = _parse_agent_models(["some-model"], [AgentType.CLAUDE_CODE, AgentType.CODEX])
        assert mapping == {AgentType.CLAUDE_CODE: "some-model", AgentType.CODEX: "some-model"}

    def test_scoped_spec_wins_over_bare(self):
        mapping = _parse_agent_models(["claude-code=haiku", "fallback"], [AgentType.CLAUDE_CODE, AgentType.CODEX])
        assert mapping[AgentType.CLAUDE_CODE] == "haiku"
        assert mapping[AgentType.CODEX] == "fallback"

    def test_unknown_agent_exits(self):
        with pytest.raises(Exception):
            _parse_agent_models(["nonsense=model"], [AgentType.CODEX])


class TestHarnessResilience:
    def test_base_url_env_injection(self, tmp_path):
        harness = ClaudeCodeHarness(tmp_path, base_url="https://example.test/v1")
        env = harness._build_env()
        assert env["ANTHROPIC_BASE_URL"] == "https://example.test/v1"
        for cls in (CodexHarness, OpenCodeHarness):
            env = cls(tmp_path, base_url="https://example.test/v1")._build_env()
            assert env["OPENAI_BASE_URL"] == "https://example.test/v1"

    def test_no_base_url_keeps_inherited_env(self, tmp_path):
        assert ClaudeCodeHarness(tmp_path)._build_env() is None

    def test_get_harness_passes_options(self, tmp_path):
        harness = get_harness(AgentType.CODEX, tmp_path, "gpt-5-mini", base_url="https://x.test", timeout=42)
        assert harness.model == "gpt-5-mini"
        assert harness.base_url == "https://x.test"
        assert harness.timeout == 42

    def test_retry_on_nonzero_exit_records_exit_code_and_retries(self, tmp_path):
        harness = CodexHarness(tmp_path, max_retries=1)
        attempts = []

        def fake_run(cmd, **kwargs):
            attempts.append(cmd)
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="boom")

        with (
            patch("agent_skill_eval.harnesses.subprocess.run", side_effect=fake_run),
            patch("agent_skill_eval.harnesses.time.sleep"),
        ):
            _, timing, _, stderr = harness.run("prompt", tmp_path / "out")

        agent_calls = [c for c in attempts if c and c[0] == "codex"]
        assert len(agent_calls) == 2
        assert timing.exit_code == 1
        assert timing.retries == 1
        assert timing.timed_out is False

    def test_timeout_recorded(self, tmp_path):
        harness = CodexHarness(tmp_path, timeout=1, max_retries=0)

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 1, output=b"partial", stderr=b"")

        with patch("agent_skill_eval.harnesses.subprocess.run", side_effect=fake_run):
            _, timing, _, stderr = harness.run("prompt", tmp_path / "out")

        assert timing.timed_out is True
        assert timing.exit_code is None
        assert "timed out" in stderr

    def test_success_no_retry(self, tmp_path):
        harness = CodexHarness(tmp_path, max_retries=3)

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        with patch("agent_skill_eval.harnesses.subprocess.run", side_effect=fake_run) as mocked:
            _, timing, _, _ = harness.run("prompt", tmp_path / "out")

        agent_calls = [c for c in mocked.call_args_list if c.args[0] and c.args[0][0] == "codex"]
        assert len(agent_calls) == 1
        assert timing.exit_code == 0
        assert timing.retries == 0


class TestValidateCommand:
    def test_valid_suite(self):
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(REPO_ROOT / "examples/write-release-notes/evals/evals.json")])
        assert result.exit_code == 0, result.output
        assert "Valid" in result.output

    def test_invalid_schema_fails(self, tmp_path):
        bad = tmp_path / "evals.json"
        bad.write_text(json.dumps({"skill_name": "x", "evals": [{"id": 1}]}))
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(bad)])
        assert result.exit_code == 1
        assert "Schema validation failed" in result.output

    def test_missing_fixture_file_fails(self, tmp_path):
        bad = tmp_path / "evals.json"
        bad.write_text(
            json.dumps(
                {
                    "skill_name": "x",
                    "evals": [
                        {"id": 1, "prompt": "p", "expected_output": "e", "files": ["files/nope.txt"]},
                    ],
                }
            )
        )
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(bad)])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_unsafe_fixture_path_fails(self, tmp_path):
        bad = tmp_path / "evals.json"
        bad.write_text(
            json.dumps(
                {
                    "skill_name": "x",
                    "evals": [
                        {"id": 1, "prompt": "p", "expected_output": "e", "files": ["../secret.txt"]},
                    ],
                }
            )
        )
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(bad)])
        assert result.exit_code == 1
        assert "Fixture path problems" in result.output
        assert "stay inside the evals directory" in result.output

    def test_fixture_workspace_target_collision_fails(self, tmp_path):
        files = tmp_path / "files"
        (files / "prod").mkdir(parents=True)
        (files / "config.json").write_text("{}")
        (files / "prod" / "config.json").write_text("{}")
        bad = tmp_path / "evals.json"
        bad.write_text(
            json.dumps(
                {
                    "skill_name": "x",
                    "evals": [
                        {
                            "id": 1,
                            "prompt": "p",
                            "expected_output": "e",
                            "files": ["files/prod/config.json", "files/config.json"],
                        },
                    ],
                }
            )
        )
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(bad)])
        assert result.exit_code == 1
        assert "workspace fixture target" in result.output
        assert "config.json" in result.output

    def test_duplicate_ids_fail(self, tmp_path):
        bad = tmp_path / "evals.json"
        bad.write_text(
            json.dumps(
                {
                    "skill_name": "x",
                    "evals": [
                        {"id": 1, "prompt": "p", "expected_output": "e"},
                        {"id": 1, "prompt": "q", "expected_output": "e"},
                    ],
                }
            )
        )
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(bad)])
        assert result.exit_code == 1
        assert "Duplicate" in result.output

    def test_structured_assertion_missing_required_field_fails(self, tmp_path):
        bad = tmp_path / "evals.json"
        bad.write_text(
            json.dumps(
                {
                    "skill_name": "x",
                    "evals": [
                        {
                            "id": "bad-assertion",
                            "prompt": "p",
                            "expected_output": "e",
                            "assertions": [{"type": "command_ran"}],
                        }
                    ],
                }
            )
        )
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(bad)])
        assert result.exit_code == 1
        assert "Structured assertion problems" in result.output
        assert "missing required field" in result.output
        assert "command" in result.output

    def test_negative_structured_assertions_missing_required_field_fail(self, tmp_path):
        bad = tmp_path / "evals.json"
        bad.write_text(
            json.dumps(
                {
                    "skill_name": "x",
                    "evals": [
                        {
                            "id": "bad-negative-assertion",
                            "prompt": "p",
                            "expected_output": "e",
                            "assertions": [
                                {"type": "file_not_contains"},
                                {"type": "json_path_not_exists"},
                            ],
                        }
                    ],
                }
            )
        )
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(bad)])
        assert result.exit_code == 1
        assert "Structured assertion problems" in result.output
        assert "contains" in result.output
        assert "json_path" in result.output

    def test_structured_assertion_rejects_unsafe_path(self, tmp_path):
        bad = tmp_path / "evals.json"
        bad.write_text(
            json.dumps(
                {
                    "skill_name": "x",
                    "evals": [
                        {
                            "id": "bad-path",
                            "prompt": "p",
                            "expected_output": "e",
                            "assertions": [{"type": "file_exists", "path": "../secret.txt"}],
                        }
                    ],
                }
            )
        )
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(bad)])
        assert result.exit_code == 1
        assert "path must be relative" in result.output
        assert "inside the workspace" in result.output


class TestListCommand:
    def test_lists_repo_examples(self):
        runner = CliRunner()
        result = runner.invoke(app, ["list", "--root", str(REPO_ROOT)])
        assert result.exit_code == 0, result.output
        assert "write-release-notes" in result.output
        assert "commit-push-pr" in result.output


class TestInitScaffold:
    def test_init_creates_negative_control_and_skill_template(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["init", "my-skill", "--output", str(tmp_path)])
        assert result.exit_code == 0, result.output

        evals = json.loads((tmp_path / "my-skill" / "evals" / "evals.json").read_text())
        suite = EvalSuite(**evals)
        ids = [str(e.id) for e in suite.evals]
        assert "negative-control" in ids
        negative = next(e for e in suite.evals if str(e.id) == "negative-control")
        assert negative.should_trigger is False

        skill_md = (tmp_path / "my-skill" / "SKILL.md").read_text()
        assert skill_md.startswith("---")
        assert "name: my-skill" in skill_md

    @pytest.mark.parametrize("template", ["basic", "review-diff", "commit-pr", "config-validator"])
    def test_init_templates_create_valid_evals(self, tmp_path, template):
        runner = CliRunner()
        result = runner.invoke(app, ["init", "my-skill", "--output", str(tmp_path), "--template", template])
        assert result.exit_code == 0, result.output

        evals_path = tmp_path / "my-skill" / "evals" / "evals.json"
        validate = runner.invoke(app, ["validate", str(evals_path)])
        assert validate.exit_code == 0, validate.output

    def test_init_plugin_layout_creates_plugin_skill_tree(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "init",
                "my-skill",
                "--output",
                str(tmp_path),
                "--layout",
                "plugin",
                "--plugin",
                "demo-plugin",
            ],
        )
        assert result.exit_code == 0, result.output

        skill_dir = tmp_path / "plugins" / "demo-plugin" / "skills" / "my-skill"
        evals_path = skill_dir / "evals" / "evals.json"
        assert evals_path.exists()
        assert (skill_dir / "SKILL.md").exists()
        validate = runner.invoke(app, ["validate-all", "--root", str(tmp_path), "--format", "json"])
        assert validate.exit_code == 0, validate.output
        payload = json.loads(validate.output)
        assert payload["suites"][0]["pair"]["skill_dir"] == "plugins/demo-plugin/skills/my-skill"

    def test_init_plugin_layout_requires_plugin_name(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["init", "my-skill", "--output", str(tmp_path), "--layout", "plugin"])
        assert result.exit_code == 1
        assert "--plugin is required" in result.output

    def test_commit_pr_template_marks_live_external_cases(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["init", "ship-it", "--output", str(tmp_path), "--template", "commit-pr"])
        assert result.exit_code == 0, result.output

        evals = json.loads((tmp_path / "ship-it" / "evals" / "evals.json").read_text())
        levels = {case["id"]: case.get("side_effect_level") for case in evals["evals"]}
        assert levels["explicit-commit-pr"] == "live-external"
        assert levels["natural-trigger"] == "live-external"
        assert levels["status-only-negative-control"] == "static"
        assert (tmp_path / "ship-it" / "evals" / "files" / "sample_change.py").exists()


class TestCompareCommand:
    def _write_benchmark(self, iter_dir: Path, pass_rate: float) -> None:
        iter_dir.mkdir(parents=True)
        (iter_dir / "benchmark.json").write_text(
            json.dumps(
                {
                    "run_summary": {
                        "fake_with_skill": {
                            "pass_rate": {"mean": pass_rate, "stddev": 0.0},
                            "time_seconds": {"mean": 1.0, "stddev": 0.0},
                            "tokens": {"mean": 1.0, "stddev": 0.0},
                        }
                    },
                    "deltas": {},
                    "delta": None,
                }
            )
        )

    def test_compare_two_iterations(self, tmp_path):
        self._write_benchmark(tmp_path / "iteration-1", 0.5)
        self._write_benchmark(tmp_path / "iteration-2", 0.75)
        runner = CliRunner()
        result = runner.invoke(app, ["compare", "--workspace", str(tmp_path), "1", "2"])
        assert result.exit_code == 0, result.output
        assert "+25.0%" in result.output

    def test_missing_iteration_fails(self, tmp_path):
        self._write_benchmark(tmp_path / "iteration-1", 0.5)
        runner = CliRunner()
        result = runner.invoke(app, ["compare", "--workspace", str(tmp_path), "1", "2"])
        assert result.exit_code == 1


class TestSchemaInSync:
    def test_checked_in_schema_matches_models(self):
        schema_path = REPO_ROOT / "schemas" / "evals.schema.json"
        checked_in = json.loads(schema_path.read_text())
        generated = EvalSuite.model_json_schema()
        generated["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        generated["title"] = "agent-skill-eval eval suite"
        assert checked_in == generated, "Regenerate schemas/evals.schema.json from EvalSuite.model_json_schema()"


class TestListDiscovery:
    def test_list_recurses_into_plugin_style_skill_layouts(self, tmp_path):
        skill_dir = tmp_path / "plugins" / "demo-plugin" / "skills" / "readme-updater"
        evals_dir = skill_dir / "evals"
        evals_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: readme-updater\ndescription: update readmes\n---\nbody",
            encoding="utf-8",
        )
        (evals_dir / "evals.json").write_text(
            json.dumps(
                {
                    "skill_name": "readme-updater",
                    "evals": [{"id": "static", "prompt": "p", "expected_output": "ok"}],
                }
            ),
            encoding="utf-8",
        )
        ignored = tmp_path / ".venv" / "skills" / "ignored"
        ignored.mkdir(parents=True)
        (ignored / "SKILL.md").write_text("---\nname: ignored\ndescription: ignored\n---\nbody", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(app, ["list", "--root", str(tmp_path)])

        assert _discover_eval_suites(tmp_path) == [evals_dir / "evals.json"]
        assert _discover_skills(tmp_path) == [skill_dir / "SKILL.md"]
        assert result.exit_code == 0, result.output
        assert "readme-updater" in result.output
        assert "plugins/demo-plugin/skills/readme-updater" in result.output
        assert "ignored" not in result.output

        json_result = runner.invoke(app, ["list", "--root", str(tmp_path), "--format", "json"])
        assert json_result.exit_code == 0, json_result.output
        payload = json.loads(json_result.output)
        assert payload["eval_suites"] == [
            {
                "skill": "readme-updater",
                "path": "plugins/demo-plugin/skills/readme-updater/evals/evals.json",
                "cases": 1,
                "error": None,
            }
        ]
        assert payload["skills"] == ["plugins/demo-plugin/skills/readme-updater"]
        assert payload["pairs"] == [
            {
                "status": "paired",
                "skill": "readme-updater",
                "skill_dir": "plugins/demo-plugin/skills/readme-updater",
                "evals_path": "plugins/demo-plugin/skills/readme-updater/evals/evals.json",
                "problems": [],
            }
        ]


class TestPortfolioReport:
    def _write_skill(self, root: Path, name: str, *, with_evals: bool = True) -> Path:
        skill_dir = root / "plugins" / "demo-plugin" / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name} skill\n---\nUse this skill for useful work.",
            encoding="utf-8",
        )
        if with_evals:
            evals_dir = skill_dir / "evals"
            evals_dir.mkdir()
            (evals_dir / "evals.json").write_text(
                json.dumps(
                    {
                        "skill_name": name,
                        "evals": [
                            {
                                "id": "static",
                                "prompt": "p",
                                "expected_output": "ok",
                                "side_effect_level": "static",
                                "assertions": ["The output contains `ok`"],
                            },
                            {
                                "id": "local",
                                "prompt": "p",
                                "expected_output": "ok",
                                "side_effect_level": "local-only",
                                "assertions": ["The response is high quality"],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
        return skill_dir

    def test_portfolio_json_reports_coverage_size_and_latest_run(self, tmp_path):
        repo = tmp_path / "repo"
        self._write_skill(repo, "readme-updater")
        self._write_skill(repo, "orphan", with_evals=False)

        workspace = tmp_path / "workspace"
        latest = workspace / "readme-updater-workspace" / "iteration-2"
        latest.mkdir(parents=True)
        (latest / "summary.json").write_text(
            json.dumps(
                {
                    "runs_completed": 2,
                    "runs_failed_grading": 0,
                    "errors": [],
                    "pass_rates": {"fake_with_skill": 1.0},
                    "stability": {"flaky_count": 0},
                    "tokens": {"total": 10},
                    "cost": {"total_usd": 0.0},
                }
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["portfolio", "--root", str(repo), "--workspace", str(workspace), "--format", "json"],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["skills"] == 2
        assert payload["paired"] == 1
        assert payload["missing_evals"] == 1
        rows = {row["skill"]: row for row in payload["rows"]}
        readme = rows["readme-updater"]
        assert readme["eval_status"] == "valid"
        assert readme["eval_cases"] == 2
        assert readme["assertions"] == 2
        assert readme["llm_assertions"] == 1
        assert readme["side_effect_counts"] == {"static": 1, "local-only": 1}
        assert readme["skill_size"]["files"] == 1
        assert readme["latest_run"]["iteration"] == 2
        assert readme["pass_rates"] == {"fake_with_skill": 1.0}
        assert rows["orphan"]["status"] == "missing-evals"
        assert rows["orphan"]["next"] == "Add evals"

    def test_portfolio_table_renders(self, tmp_path):
        repo = tmp_path / "repo"
        self._write_skill(repo, "readme-updater")

        runner = CliRunner()
        result = runner.invoke(app, ["portfolio", "--root", str(repo)])

        assert result.exit_code == 0, result.output
        assert "Skill Portfolio" in result.output
        assert "readme-updater" in result.output
        assert "Tighten assertions" in result.output


class TestValidateAll:
    def test_validate_all_recurses_into_plugin_style_layouts(self, tmp_path):
        skill_dir = tmp_path / "plugins" / "demo-plugin" / "skills" / "readme-updater"
        evals_dir = skill_dir / "evals"
        evals_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: readme-updater\ndescription: update readmes\n---\nbody",
            encoding="utf-8",
        )
        (evals_dir / "evals.json").write_text(
            json.dumps(
                {
                    "skill_name": "readme-updater",
                    "evals": [
                        {
                            "id": "static",
                            "prompt": "p",
                            "expected_output": "ok",
                            "side_effect_level": "static",
                            "assertions": ["The output contains `ok`"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(app, ["validate-all", "--root", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "readme-updater" in result.output
        assert "All 1 eval suite(s) valid" in result.output

        json_result = runner.invoke(app, ["validate-all", "--root", str(tmp_path), "--format", "json"])
        assert json_result.exit_code == 0, json_result.output
        payload = json.loads(json_result.output)
        assert payload["failures"] == 0
        assert payload["suites"][0]["skill"] == "readme-updater"
        assert payload["suites"][0]["path"] == "plugins/demo-plugin/skills/readme-updater/evals/evals.json"
        assert payload["suites"][0]["pair"]["status"] == "paired"

    def test_validate_all_strict_flags_low_signal_suites(self, tmp_path):
        skill_dir = tmp_path / "plugins" / "demo-plugin" / "skills" / "weak-skill"
        evals_dir = skill_dir / "evals"
        evals_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: weak-skill\ndescription: weak suite\n---\nbody",
            encoding="utf-8",
        )
        (evals_dir / "evals.json").write_text(
            json.dumps(
                {
                    "skill_name": "weak-skill",
                    "evals": [
                        {
                            "id": "happy",
                            "prompt": "p",
                            "expected_output": "ok",
                            "side_effect_level": "static",
                            "force_skill_invocation": True,
                            "assertions": ["The agent produces a good answer"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(app, ["validate-all", "--root", str(tmp_path), "--strict", "--format", "json"])

        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["failures"] == 1
        assert payload["suites"][0]["status"] == "strict_fail"
        assert any(
            warning.startswith("all evals force skill invocation")
            for warning in payload["suites"][0]["strict_warnings"]
        )
        assert "suite has no negative or missing-input control case" in payload["suites"][0]["strict_warnings"]

    def test_validate_all_fails_when_any_suite_is_invalid(self, tmp_path):
        evals_dir = tmp_path / "plugins" / "demo-plugin" / "skills" / "bad-skill" / "evals"
        evals_dir.mkdir(parents=True)
        (evals_dir / "evals.json").write_text(
            json.dumps(
                {
                    "skill_name": "bad-skill",
                    "evals": [
                        {"id": "same", "prompt": "p", "expected_output": "ok"},
                        {"id": "same", "prompt": "p", "expected_output": "ok"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(app, ["validate-all", "--root", str(tmp_path)])

        assert result.exit_code == 1
        assert "Duplicate eval ids:" in result.output
        assert "['same']" in result.output

    def test_validate_all_fails_plugin_style_suite_without_skill_md(self, tmp_path):
        evals_dir = tmp_path / "plugins" / "demo-plugin" / "skills" / "orphan-skill" / "evals"
        evals_dir.mkdir(parents=True)
        (evals_dir / "evals.json").write_text(
            json.dumps(
                {
                    "skill_name": "orphan-skill",
                    "evals": [{"id": "static", "prompt": "p", "expected_output": "ok"}],
                }
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(app, ["validate-all", "--root", str(tmp_path)])

        assert result.exit_code == 1
        assert "Missing SKILL.md" in result.output
        assert "plugin-style eval suite" in result.output


class TestRunsLayout:
    def test_multi_run_fake_harness_produces_run_dirs_and_pass_at_k(self, tmp_path):
        runner = CliRunner()
        workspace = tmp_path / "ws"
        result = runner.invoke(
            app,
            [
                "run",
                "--skill",
                str(FIXTURES / "skills" / "format-json"),
                "--evals",
                str(FIXTURES / "evals" / "format-json.json"),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
                "--runs",
                "2",
                "--no-baseline",
            ],
        )
        assert result.exit_code == 0, result.output

        iteration_dir = workspace / "format-json-workspace" / "iteration-1"
        config_dir = iteration_dir / "eval-explicit-invoke" / "fake" / "with_skill"
        assert (config_dir / "run-1" / "outputs" / "output.txt").exists()
        assert (config_dir / "run-2" / "outputs" / "output.txt").exists()

        benchmark = json.loads((iteration_dir / "benchmark.json").read_text())
        stats = benchmark["run_summary"]["fake_with_skill"]
        assert stats["k"] == 2
        assert 0.0 <= stats["pass_at_k"] <= 1.0


class TestAssertionRouting:
    """Regression tests for assertion-routing bugs found by live smoke runs."""

    def test_prominently_does_not_route_to_pr_check(self, tmp_path, grader):
        result = grader._check_assertion(
            "The breaking change is mentioned prominently and includes migration guidance",
            tmp_path,
            "agent output",
            None,
            True,
            {},
        )
        # Not deterministically checkable -> falls through to the LLM.
        assert result.method == "unknown"

    def test_present_does_not_route_to_pr_check(self, tmp_path, grader):
        result = grader._check_assertion(
            "The notes do not mention any change that is not present in commits.txt",
            tmp_path,
            "agent output",
            None,
            True,
            {},
        )
        assert result.method == "unknown"

    def test_pull_request_still_routes_to_pr_check(self, tmp_path, grader):
        result = grader._check_assertion("A pull request was created", tmp_path, "", None, True, {})
        assert "PR" in result.evidence or "pr" in result.evidence.lower()

    def test_bare_pr_word_still_routes_to_pr_check(self, tmp_path, grader):
        result = grader._check_assertion("A PR was opened for the change", tmp_path, "", None, True, {})
        assert result.method == "deterministic"
        assert result.passed is False

    def test_file_backtick_exists_routes_to_file_check(self, tmp_path, grader):
        (tmp_path / "RELEASE_NOTES.md").write_text("# notes")
        result = grader._check_assertion(
            "The file `RELEASE_NOTES.md` exists",
            tmp_path,
            "",
            tmp_path,
            True,
            {},
        )
        assert result.passed is True
        assert "RELEASE_NOTES.md" in result.evidence


class TestLLMGraderWorkspaceContext:
    def test_small_workspace_files_are_inlined_and_skill_dirs_excluded(self, tmp_path):
        from agent_skill_eval.graders import LLMGrader

        (tmp_path / "RELEASE_NOTES.md").write_text("# Release Notes\n\n## Features\n- thing")
        skill_dir = tmp_path / ".claude" / "skills" / "demo"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("secret skill text")
        (tmp_path / "big.bin").write_bytes(b"x" * 10_000)

        grader = LLMGrader()
        contents = grader._read_workspace_files(tmp_path)
        assert "RELEASE_NOTES.md" in contents
        assert "## Features" in contents
        assert "secret skill text" not in contents
        assert "big.bin" not in contents

        listing = grader._list_workspace_files(tmp_path)
        assert "RELEASE_NOTES.md" in listing
        assert "SKILL.md" not in listing


class TestRetryWorkspaceGuard:
    class _ScriptHarness(CodexHarness):
        """Harness whose 'agent' is a shell script, so the retry loop runs
        for real while git fingerprinting hits the actual workspace."""

        script: str = "exit 1"

        def build_command(self, prompt, output_dir):
            return ["bash", "-c", self.script]

    def test_no_retry_when_failed_attempt_mutated_workspace(self, git_workspace, tmp_path):
        harness = self._ScriptHarness(git_workspace, max_retries=2)
        harness.script = "echo mutated > newfile.txt; exit 1"

        with patch("agent_skill_eval.harnesses.time.sleep"):
            _, timing, _, stderr = harness.run("prompt", tmp_path / "out")

        assert timing.exit_code == 1
        assert timing.retries == 0
        assert "not retrying" in stderr
        # Only one attempt ran: the mutation marker exists exactly once.
        assert (git_workspace / "newfile.txt").read_text() == "mutated\n"

    def test_retry_proceeds_when_workspace_clean(self, git_workspace, tmp_path):
        harness = self._ScriptHarness(git_workspace, max_retries=2)
        harness.script = "exit 1"

        with patch("agent_skill_eval.harnesses.time.sleep"):
            _, timing, _, stderr = harness.run("prompt", tmp_path / "out")

        assert timing.exit_code == 1
        assert timing.retries == 2
        assert "not retrying" not in stderr


class TestEmptyAgentModelSpec:
    def test_blank_bare_spec_rejected(self):
        with pytest.raises(Exception):
            _parse_agent_models(["  "], [AgentType.CODEX])
