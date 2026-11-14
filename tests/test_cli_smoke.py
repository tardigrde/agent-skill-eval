from __future__ import annotations

import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from typer.testing import CliRunner

from agent_skill_eval.cli import app

FIXTURES = Path(__file__).parent / "fixtures"


class TestSkillEvalCliSmoke:
    def _git(self, repo: Path, *args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)

    def _init_git_repo(self, repo: Path) -> None:
        self._git(repo, "init", "-b", "main")
        self._git(repo, "config", "user.email", "test@example.com")
        self._git(repo, "config", "user.name", "Test User")

    def _commit_all(self, repo: Path, message: str) -> None:
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-m", message)

    def _write_skill(
        self,
        root: Path,
        name: str,
        *,
        prompt: str = 'Format {"status":"formatted-json-ok"}.',
        assertion: str = "The output contains `formatted-json-ok`",
        side_effect_level: str = "static",
    ) -> None:
        skill_dir = root / "plugins" / "demo" / "skills" / name
        evals_dir = skill_dir / "evals"
        evals_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Format JSON\n---\nbody",
            encoding="utf-8",
        )
        (evals_dir / "evals.json").write_text(
            json.dumps(
                {
                    "skill_name": name,
                    "evals": [
                        {
                            "id": "formats-json",
                            "prompt": prompt,
                            "expected_output": "formatted JSON",
                            "side_effect_level": side_effect_level,
                            "assertions": [assertion],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_run_report_and_init_with_fake_harness(self, tmp_path):
        runner = CliRunner()
        workspace = tmp_path / "workspace"
        skill = FIXTURES / "skills" / "format-json"
        evals = FIXTURES / "evals" / "format-json.json"

        run_result = runner.invoke(
            app,
            [
                "run",
                "--skill",
                str(skill),
                "--evals",
                str(evals),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
                "--cleanup",
            ],
        )

        assert run_result.exit_code == 0, run_result.output

        iteration_dir = workspace / "format-json-workspace" / "iteration-1"
        assert (iteration_dir / "benchmark.json").exists()
        assert (iteration_dir / "cleanup.json").exists()

        benchmark = json.loads((iteration_dir / "benchmark.json").read_text())
        assert "fake_with_skill" in benchmark["run_summary"]
        assert "fake_without_skill" in benchmark["run_summary"]

        eval_ids = ["explicit-invoke", "negative-control", "content-contains"]
        for eval_id in eval_ids:
            for config in ["with_skill", "without_skill"]:
                config_dir = iteration_dir / f"eval-{eval_id}" / "fake" / config
                assert (config_dir / "grading.json").exists(), f"missing grading for {eval_id}/{config}"
                assert (config_dir / "run_meta.json").exists(), f"missing run_meta for {eval_id}/{config}"
                assert (config_dir / "trajectory.json").exists(), f"missing trajectory for {eval_id}/{config}"
                assert (config_dir / "timing.json").exists(), f"missing timing for {eval_id}/{config}"
                assert (config_dir / "outputs" / "output.txt").exists(), f"missing output for {eval_id}/{config}"
                assert (config_dir / "outputs" / "pre_state.json").exists()
                assert (config_dir / "outputs" / "post_state.json").exists()
                meta = json.loads((config_dir / "run_meta.json").read_text())
                assert meta["skill_hash"]
                assert meta["side_effect_level"] == "local-only"
                trajectory = json.loads((config_dir / "trajectory.json").read_text())
                assert any(event["event"] == "agent_finished" for event in trajectory)

        summary = json.loads((iteration_dir / "summary.json").read_text())
        assert summary["skill_hash"]
        assert summary["side_effect_levels"] == ["local-only"]

        report_result = runner.invoke(
            app,
            [
                "report",
                "--workspace",
                str(workspace / "format-json-workspace"),
            ],
        )

        assert report_result.exit_code == 0, report_result.output
        assert "fake/with_skill" in report_result.output
        assert "fake/without_skill" in report_result.output

        # The rich table truncates long config names at narrow widths, so
        # assert the full names via the markdown format.
        md_result = runner.invoke(
            app,
            [
                "report",
                "--workspace",
                str(workspace / "format-json-workspace"),
                "--format",
                "markdown",
            ],
        )
        assert md_result.exit_code == 0, md_result.output
        assert "fake_with_skill" in md_result.output
        assert "fake_without_skill" in md_result.output

        html_result = runner.invoke(
            app,
            [
                "report",
                "--workspace",
                str(workspace / "format-json-workspace"),
                "--format",
                "html",
            ],
        )
        assert html_result.exit_code == 0, html_result.output
        assert "<!doctype html>" in html_result.output
        assert "fake_with_skill" in html_result.output

        init_output = tmp_path / "init-output"
        init_result = runner.invoke(app, ["init", "format-json", "--output", str(init_output)])

        assert init_result.exit_code == 0, init_result.output
        assert (init_output / "format-json" / "evals" / "evals.json").exists()

    def test_run_all_runs_paired_suites_and_skips_filtered_levels(self, tmp_path):
        repo = tmp_path / "repo"
        self._write_skill(repo, "format-json", side_effect_level="static")
        self._write_skill(repo, "format-json-local", side_effect_level="local-only")

        workspace = tmp_path / "workspace"
        junit_path = workspace / "run-all.xml"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run-all",
                "--root",
                str(repo),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
                "--no-baseline",
                "--side-effect-level",
                "static",
                "--junit-xml",
                str(junit_path),
            ],
        )

        assert result.exit_code == 0, result.output
        manifest = json.loads((workspace / "run-all-iteration-1.json").read_text())
        statuses = {suite["skill"]: suite["status"] for suite in manifest["suites"]}
        assert statuses == {"format-json": "ok", "format-json-local": "skipped"}
        assert manifest["failures"] == 0
        assert manifest["runs_failed_grading"] == 0
        assert manifest["suites_failed_grading"] == 0
        assert manifest["clean"] is True
        assert manifest["agents"] == ["fake"]
        assert manifest["agent_models"] == {}
        assert manifest["baseline"] is False
        assert manifest["grader_model"] == ""
        assert manifest["skipped"] == 1
        assert manifest["duration_seconds"] >= 0
        suite = next(suite for suite in manifest["suites"] if suite["skill"] == "format-json")
        assert suite["summary"]["runs_completed"] == 1
        assert suite["summary"]["runs_failed_grading"] == 0
        assert suite["summary"]["tokens"]["total"] == 1
        assert suite["summary"]["cost"]["total_usd"] == 0.0
        assert suite["summary"]["pass_rates"]["fake_with_skill"] == 1.0
        assert (workspace / "format-json-workspace" / "iteration-1" / "summary.json").exists()

    def test_run_all_can_run_suites_concurrently(self, tmp_path):
        repo = tmp_path / "repo"
        self._write_skill(repo, "format-json", side_effect_level="static")
        self._write_skill(repo, "other-json", side_effect_level="static")

        workspace = tmp_path / "workspace"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run-all",
                "--root",
                str(repo),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
                "--no-baseline",
                "--side-effect-level",
                "static",
                "--suite-concurrency",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        manifest = json.loads((workspace / "run-all-iteration-1.json").read_text())
        assert manifest["suite_concurrency"] == 2
        assert [suite["skill"] for suite in manifest["suites"]] == ["format-json", "other-json"]
        assert {suite["status"] for suite in manifest["suites"]} == {"ok"}
        assert (workspace / "format-json-workspace" / "iteration-1" / "summary.json").exists()
        assert (workspace / "other-json-workspace" / "iteration-1" / "summary.json").exists()

    def test_run_all_plan_only_writes_manifest_without_running_agents(self, tmp_path):
        repo = tmp_path / "repo"
        self._write_skill(repo, "format-json", side_effect_level="static")
        self._write_skill(repo, "format-json-local", side_effect_level="local-only")

        workspace = tmp_path / "workspace"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run-all",
                "--root",
                str(repo),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
                "--no-baseline",
                "--side-effect-level",
                "static",
                "--plan-only",
            ],
        )

        assert result.exit_code == 0, result.output
        manifest = json.loads((workspace / "run-all-iteration-1.json").read_text())
        assert manifest["plan_only"] is True
        statuses = {suite["skill"]: suite["status"] for suite in manifest["suites"]}
        assert statuses == {"format-json": "planned", "format-json-local": "skipped"}
        assert manifest["planned_suites"] == 1
        assert manifest["planned_cases"] == 1
        planned = next(suite for suite in manifest["suites"] if suite["skill"] == "format-json")
        assert planned["planned_cases"] == 1
        assert manifest["clean"] is True
        assert not (workspace / "format-json-workspace" / "iteration-1" / "summary.json").exists()

        report = runner.invoke(
            app,
            [
                "report-all",
                "--workspace",
                str(workspace),
                "--format",
                "markdown",
            ],
        )
        assert report.exit_code == 0, report.output
        assert "planned_suites=1" in report.output
        assert "planned_cases=1" in report.output
        assert "| Skill | Status | Cases | Completed |" in report.output
        assert "| format-json | planned | 1 | 0 |" in report.output

    def test_run_all_records_failed_assertions_without_execution_failure(self, tmp_path):
        repo = tmp_path / "repo"
        self._write_skill(
            repo,
            "format-json",
            prompt="Return plain text.",
            assertion="The output contains `never-present`",
        )

        workspace = tmp_path / "workspace"
        junit_path = workspace / "run-all.xml"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run-all",
                "--root",
                str(repo),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
                "--no-baseline",
                "--side-effect-level",
                "static",
                "--junit-xml",
                str(junit_path),
            ],
        )

        assert result.exit_code == 0, result.output
        manifest = json.loads((workspace / "run-all-iteration-1.json").read_text())
        assert manifest["failures"] == 0
        assert manifest["runs_failed_grading"] == 1
        assert manifest["suites_failed_grading"] == 1
        assert manifest["clean"] is False
        assert manifest["suites"][0]["status"] == "failed_assertions"
        assert manifest["suites"][0]["summary"]["runs_failed_grading"] == 1
        assert "failed assertions" in result.output
        junit = ET.parse(junit_path).getroot()
        assert junit.tag == "testsuite"
        assert junit.attrib["tests"] == "1"
        assert junit.attrib["failures"] == "1"
        testcase = junit.find("testcase")
        assert testcase is not None
        assert testcase.attrib["name"] == "format-json"
        failure = testcase.find("failure")
        assert failure is not None
        assert "never-present" in (failure.text or "")

        markdown = runner.invoke(
            app,
            [
                "report-all",
                "--workspace",
                str(workspace),
                "--format",
                "markdown",
            ],
        )
        assert markdown.exit_code == 0, markdown.output
        assert "format-json" in markdown.output
        assert "Tokens" in markdown.output
        assert "Cost" in markdown.output
        assert "failed_assertions" in markdown.output
        assert "never-present" in markdown.output

        report_json = runner.invoke(
            app,
            [
                "report-all",
                "--workspace",
                str(workspace),
                "--format",
                "json",
            ],
        )
        assert report_json.exit_code == 0, report_json.output
        assert json.loads(report_json.output)["clean"] is False

        rerun = runner.invoke(
            app,
            [
                "rerun-failed",
                "--from",
                str(workspace / "run-all-iteration-1.json"),
                "--workspace",
                str(tmp_path / "reruns"),
                "--format",
                "shell",
            ],
        )
        assert rerun.exit_code == 0, rerun.output
        assert "agent-skill-eval run" in rerun.output
        assert "--skill" in rerun.output
        assert "--eval-id formats-json" in rerun.output
        assert "--grader-model ''" in rerun.output
        assert "--no-baseline" in rerun.output

    def test_run_all_can_fail_on_assertions_for_ci(self, tmp_path):
        repo = tmp_path / "repo"
        self._write_skill(
            repo,
            "format-json",
            prompt="Return plain text.",
            assertion="The output contains `never-present`",
        )

        workspace = tmp_path / "workspace"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run-all",
                "--root",
                str(repo),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
                "--no-baseline",
                "--side-effect-level",
                "static",
                "--fail-on-assertions",
            ],
        )

        assert result.exit_code == 1
        manifest = json.loads((workspace / "run-all-iteration-1.json").read_text())
        assert manifest["clean"] is False

    def test_run_all_can_fail_on_min_pass_rate_for_ci(self, tmp_path):
        repo = tmp_path / "repo"
        self._write_skill(
            repo,
            "format-json",
            prompt="Return plain text.",
            assertion="The output contains `never-present`",
        )

        workspace = tmp_path / "workspace"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run-all",
                "--root",
                str(repo),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
                "--no-baseline",
                "--side-effect-level",
                "static",
                "--min-pass-rate",
                "1.0",
            ],
        )

        assert result.exit_code == 1
        manifest = json.loads((workspace / "run-all-iteration-1.json").read_text())
        assert manifest["min_pass_rate"] == 1.0
        assert manifest["suites_below_min_pass_rate"] == 1
        assert manifest["clean"] is False
        assert manifest["suites"][0]["summary"]["pass_rates_below_threshold"]["fake_with_skill"] == 0.0

    def test_run_all_forwards_source_repo_and_hooks_to_each_suite(self, tmp_path):
        repo = tmp_path / "repo"
        self._write_skill(repo, "format-json")

        source_repo = tmp_path / "source-repo"
        source_repo.mkdir()
        (source_repo / "README.md").write_text("source fixture\n", encoding="utf-8")
        self._init_git_repo(source_repo)

        workspace = tmp_path / "workspace"
        hook_output = tmp_path / "pre-run-hook.txt"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run-all",
                "--root",
                str(repo),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
                "--no-baseline",
                "--side-effect-level",
                "static",
                "--source-repo",
                str(source_repo),
                "--pre-run-command",
                f'echo "$ASE_SOURCE_REPO|$ASE_SKILL_NAME" > {hook_output}',
                "--post-run-command",
                "exit 5",
            ],
        )

        assert result.exit_code == 0, result.output
        manifest = json.loads((workspace / "run-all-iteration-1.json").read_text())
        assert manifest["source_repo"] == str(source_repo)
        assert manifest["hook_counts"] == {"pre_run": 1, "post_grade": 0, "post_run": 1}
        assert hook_output.read_text().strip() == f"{source_repo}|format-json"

        iteration_dir = workspace / "format-json-workspace" / "iteration-1"
        summary = json.loads((iteration_dir / "summary.json").read_text())
        assert summary["hooks"]["post_run"][0]["exit_code"] == 5

        meta_path = iteration_dir / "eval-formats-json" / "fake" / "with_skill" / "run_meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta["source_repo"] == str(source_repo)

    def test_run_all_forwards_budget_options_to_each_suite(self, tmp_path):
        repo = tmp_path / "repo"
        self._write_skill(repo, "format-json")

        workspace = tmp_path / "workspace"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run-all",
                "--root",
                str(repo),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
                "--no-baseline",
                "--side-effect-level",
                "static",
                "--max-total-tokens-per-case",
                "0",
                "--budget-action",
                "fail",
            ],
        )

        assert result.exit_code == 0, result.output
        manifest = json.loads((workspace / "run-all-iteration-1.json").read_text())
        assert manifest["budget"]["max_total_tokens"] == 0
        assert manifest["runs_failed_grading"] == 1
        assert manifest["suites_failed_grading"] == 1
        assert manifest["suites"][0]["status"] == "failed_assertions"
        failed = manifest["suites"][0]["summary"]["failed_runs"][0]
        assert failed["failed_assertions"] == ["The run stayed within its budget"]

        summary = json.loads((workspace / "format-json-workspace" / "iteration-1" / "summary.json").read_text())
        assert summary["budget"]["exceeded_runs"][0]["reason"] == "total tokens 1 > 0"

        rerun = runner.invoke(
            app,
            [
                "rerun-failed",
                "--from",
                str(workspace / "run-all-iteration-1.json"),
            ],
        )
        assert rerun.exit_code == 0, rerun.output
        assert "--max-total-tokens-per-case 0" in rerun.output
        assert "--budget-action fail" in rerun.output
        assert "--grader-model ''" in rerun.output

    def test_run_all_changed_only_filters_to_touched_skill_suite(self, tmp_path):
        repo = tmp_path / "repo"
        self._write_skill(repo, "format-json")
        self._write_skill(repo, "other-json")
        self._init_git_repo(repo)
        self._commit_all(repo, "baseline")

        evals_path = repo / "plugins" / "demo" / "skills" / "format-json" / "evals" / "evals.json"
        payload = json.loads(evals_path.read_text())
        payload["evals"][0]["prompt"] = 'Format {"status":"formatted-json-ok", "changed": true}.'
        evals_path.write_text(json.dumps(payload), encoding="utf-8")
        self._commit_all(repo, "change one suite")

        workspace = tmp_path / "workspace"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run-all",
                "--root",
                str(repo),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
                "--no-baseline",
                "--side-effect-level",
                "static",
                "--changed-only",
                "--base-ref",
                "HEAD~1",
            ],
        )

        assert result.exit_code == 0, result.output
        manifest = json.loads((workspace / "run-all-iteration-1.json").read_text())
        assert manifest["changed_only"] is True
        assert manifest["base_ref"] == "HEAD~1"
        assert [suite["skill"] for suite in manifest["suites"]] == ["format-json"]

    def test_run_all_changed_only_no_changed_suites_writes_clean_manifest(self, tmp_path):
        repo = tmp_path / "repo"
        self._write_skill(repo, "format-json")
        (repo / "README.md").write_text("baseline\n", encoding="utf-8")
        self._init_git_repo(repo)
        self._commit_all(repo, "baseline")
        (repo / "README.md").write_text("changed\n", encoding="utf-8")
        self._commit_all(repo, "change docs")

        workspace = tmp_path / "workspace"
        junit_path = workspace / "run-all.xml"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run-all",
                "--root",
                str(repo),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
                "--no-baseline",
                "--side-effect-level",
                "static",
                "--changed-only",
                "--base-ref",
                "HEAD~1",
                "--junit-xml",
                str(junit_path),
            ],
        )

        assert result.exit_code == 0, result.output
        manifest = json.loads((workspace / "run-all-iteration-1.json").read_text())
        assert manifest["clean"] is True
        assert manifest["suites"] == []
        assert manifest["changed_paths"] == ["README.md"]
        junit = ET.parse(junit_path).getroot()
        assert junit.attrib["tests"] == "0"

    def test_run_reuses_validation_for_missing_fixtures(self, tmp_path):
        runner = CliRunner()
        workspace = tmp_path / "workspace"
        skill = FIXTURES / "skills" / "format-json"
        evals = tmp_path / "evals.json"
        evals.write_text(
            json.dumps(
                {
                    "skill_name": "format-json",
                    "evals": [
                        {
                            "id": "missing-fixture",
                            "prompt": "format this",
                            "expected_output": "ok",
                            "files": ["files/does-not-exist.json"],
                            "assertions": ["The output contains `ok`"],
                        }
                    ],
                }
            )
        )

        result = runner.invoke(
            app,
            [
                "run",
                "--skill",
                str(skill),
                "--evals",
                str(evals),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
            ],
        )

        assert result.exit_code == 1
        assert "Referenced fixture files not found" in result.output
        assert not workspace.exists()

    def test_run_reuses_validation_for_duplicate_eval_ids(self, tmp_path):
        runner = CliRunner()
        workspace = tmp_path / "workspace"
        skill = FIXTURES / "skills" / "format-json"
        evals = tmp_path / "evals.json"
        evals.write_text(
            json.dumps(
                {
                    "skill_name": "format-json",
                    "evals": [
                        {
                            "id": "same",
                            "prompt": "format this",
                            "expected_output": "ok",
                            "assertions": ["The output contains `ok`"],
                        },
                        {
                            "id": "same",
                            "prompt": "format that",
                            "expected_output": "ok",
                            "assertions": ["The output contains `ok`"],
                        },
                    ],
                }
            )
        )

        result = runner.invoke(
            app,
            [
                "run",
                "--skill",
                str(skill),
                "--evals",
                str(evals),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
            ],
        )

        assert result.exit_code == 1
        assert "Duplicate eval ids" in result.output
        assert not workspace.exists()

    def test_run_filters_by_side_effect_level(self, tmp_path):
        runner = CliRunner()
        workspace = tmp_path / "workspace"
        skill = FIXTURES / "skills" / "format-json"
        evals = tmp_path / "evals.json"
        evals.write_text(
            json.dumps(
                {
                    "skill_name": "format-json",
                    "evals": [
                        {
                            "id": "static-case",
                            "prompt": 'Format {"status":"formatted-json-ok"}.',
                            "expected_output": "ok",
                            "side_effect_level": "static",
                            "assertions": ["The output contains `formatted-json-ok`"],
                        },
                        {
                            "id": "live-case",
                            "prompt": 'Format {"status":"formatted-json-ok"}.',
                            "expected_output": "ok",
                            "side_effect_level": "live-external",
                            "assertions": ["The output contains `formatted-json-ok`"],
                        },
                    ],
                }
            )
        )

        result = runner.invoke(
            app,
            [
                "run",
                "--skill",
                str(skill),
                "--evals",
                str(evals),
                "--agent",
                "fake",
                "--workspace",
                str(workspace),
                "--grader-model",
                "",
                "--side-effect-level",
                "static",
            ],
        )

        assert result.exit_code == 0, result.output
        iteration_dir = workspace / "format-json-workspace" / "iteration-1"
        assert (iteration_dir / "eval-static-case").exists()
        assert not (iteration_dir / "eval-live-case").exists()
        meta = json.loads((iteration_dir / "evals_meta.json").read_text())
        assert meta["selected_side_effect_levels"] == ["static"]

    def test_doctor_passes_for_fake_agent_and_valid_suite(self):
        runner = CliRunner()
        skill = FIXTURES / "skills" / "format-json"
        evals = FIXTURES / "evals" / "format-json.json"

        result = runner.invoke(
            app,
            [
                "doctor",
                "--skill",
                str(skill),
                "--evals",
                str(evals),
                "--agent",
                "fake",
                "--grader-model",
                "",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Doctor checks passed" in result.output

    def test_doctor_warns_on_frontmatter_name_mismatch(self, tmp_path):
        skill = tmp_path / "dir-name"
        skill.mkdir()
        (skill / "SKILL.md").write_text("---\nname: other-name\ndescription: A test skill\n---\nbody")

        result = CliRunner().invoke(
            app,
            ["doctor", "--skill", str(skill), "--agent", "fake", "--grader-model", ""],
        )

        assert result.exit_code == 0, result.output
        assert "other-name" in result.output
        assert "Doctor found 2 warning(s)" in result.output

    def test_doctor_fails_for_invalid_suite(self, tmp_path):
        runner = CliRunner()
        skill = FIXTURES / "skills" / "format-json"
        evals = tmp_path / "evals.json"
        evals.write_text(
            json.dumps(
                {
                    "skill_name": "format-json",
                    "evals": [
                        {
                            "id": "bad",
                            "prompt": "format this",
                            "expected_output": "ok",
                            "files": ["files/missing.json"],
                        }
                    ],
                }
            )
        )

        result = runner.invoke(
            app,
            [
                "doctor",
                "--skill",
                str(skill),
                "--evals",
                str(evals),
                "--agent",
                "fake",
                "--grader-model",
                "",
            ],
        )

        assert result.exit_code == 1
        assert "Referenced fixture files not found" in result.output
