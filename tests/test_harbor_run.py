from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent_skill_eval.harbor.models import HarborCase, HarborExportManifest
from agent_skill_eval.harbor.run import HarborRunOptions, _arm_task_names, _job_config, run_harbor
from agent_skill_eval.models import AgentType

FIXTURES = Path(__file__).parent / "fixtures"
EVALS = FIXTURES / "evals" / "format-json.json"
SKILL = FIXTURES / "skills" / "format-json"

FAKE_HARBOR = """
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

args = sys.argv[1:]
config_path = Path(args[args.index("-c") + 1])
config = yaml.safe_load(config_path.read_text())
job_name = config["job_name"]
job_dir = Path(config["jobs_dir"]) / job_name
shutil.copytree(os.environ["FAKE_JOB_TEMPLATE"], job_dir)
with_skill = "without_skill" not in job_name

trial = next(path for path in job_dir.iterdir() if path.is_dir())
result = json.loads((trial / "result.json").read_text())
result["config"]["agent"]["skills"] = ["/tmp/demo"] if with_skill else []
(trial / "result.json").write_text(json.dumps(result))
if os.environ.get("FAKE_FAIL_WITHOUT") and not with_skill:
    sys.exit(3)
"""


def _write_template(tmp_path: Path) -> Path:
    job_dir = tmp_path / "template"
    trial = job_dir / "case-content-contains__abc123"
    for path in (trial / "verifier", trial / "agent", trial / "artifacts" / "app"):
        path.mkdir(parents=True, exist_ok=True)
    (trial / "artifacts" / "app" / "output.txt").write_text("Hello, world!\n")
    (trial / "agent" / "opencode-v2.txt").write_text('{"type":"text","text":"Hello, world!"}\n')

    case = {
        "case_id": "content-contains",
        "task_name": "case-content-contains",
        "workdir": "/app",
        "assertions": [
            {"type": "file_exists", "path": "output.txt"},
            "The output contains `Hello`",
        ],
        "deferred": [1],
        "expected_output": "Hello",
        "should_trigger": True,
        "force_skill_invocation": False,
        "side_effect_contract": None,
        "side_effect_level": "local-only",
        "skill_name": "demo",
        "skill_frontmatter_name": "demo",
    }
    (trial / "verifier" / "reward-details.json").write_text(
        json.dumps(
            {
                "case_id": "content-contains",
                "task_name": "case-content-contains",
                "workdir": "/app",
                "case": case,
                "assertions": [
                    {
                        "index": 0,
                        "text": "file_exists (path='output.txt')",
                        "passed": True,
                        "skipped": False,
                        "method": "structured:file_exists",
                        "evidence": "output.txt exists",
                        "deferred": False,
                    },
                    {
                        "index": 1,
                        "text": "The output contains `Hello`",
                        "passed": False,
                        "skipped": True,
                        "method": "deferred",
                        "evidence": "graded by the importer",
                        "deferred": True,
                    },
                ],
                "summary": {"passed": 1, "failed": 0, "total": 2, "pass_rate": 1.0, "skipped": 1},
                "has_git": False,
                "baseline_violations": [],
            }
        )
    )
    (trial / "agent" / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "steps": [
                    {
                        "step_id": 2,
                        "source": "agent",
                        "message": "Hello, world!",
                        "tool_calls": [
                            {
                                "tool_call_id": "c1",
                                "function_name": "skill",
                                "arguments": {"id": "skill"},
                            }
                        ],
                        "observation": {
                            "results": [
                                {
                                    "source_call_id": "c1",
                                    "content": '<skill_content name="demo">demo</skill_content>',
                                }
                            ]
                        },
                    }
                ],
            }
        )
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "skill-eval/case-content-contains",
                "trial_name": "case-content-contains__abc123",
                "config": {
                    "task": {"path": str(tmp_path / "export" / "tasks" / "case-content-contains")},
                    "agent": {
                        "name": "opencode-v2",
                        "model_name": "openrouter/deepseek/deepseek-v4.1-flash",
                        "skills": ["/tmp/demo"],
                        "kwargs": {},
                        "base_url": None,
                    },
                },
                "agent_info": {
                    "name": "opencode-v2",
                    "version": "2.3.0",
                    "model_info": {"name": "deepseek/deepseek-v4.1-flash", "provider": "openrouter"},
                },
                "agent_result": {
                    "n_input_tokens": 100,
                    "n_cache_tokens": 50,
                    "n_output_tokens": 10,
                    "cost_usd": 0.001,
                },
                "verifier_result": {"rewards": {"correctness": 1.0}},
                "reward_valid": True,
                "exception_info": None,
                "agent_execution": {
                    "started_at": "2026-11-15T00:00:00+00:00",
                    "finished_at": "2026-11-15T00:00:10+00:00",
                },
            }
        )
    )
    (job_dir / "result.json").write_text(json.dumps({"schema_version": 1}))
    return job_dir


def test_arm_task_names_keep_natural_suffixes_in_both_arms():
    manifest = HarborExportManifest(
        suite="demo",
        skill_name="demo",
        skill_hash="abc",
        evals_path="evals.json",
        out_dir="out",
        workdir="/app",
        base_image="python:3.12-slim",
        task_version="1.0",
        cases=[
            HarborCase(
                case_id="foo__without",
                task_names=["case-foo__without"],
                arm="both",
            )
        ],
        tasks={"case-foo__without": "foo__without"},
        task_arms={"case-foo__without": "both"},
    )

    assert _arm_task_names(manifest, True) == ["case-foo__without"]
    assert _arm_task_names(manifest, False) == ["case-foo__without"]


def test_arm_task_names_select_forced_variants():
    manifest = HarborExportManifest(
        suite="demo",
        skill_name="demo",
        skill_hash="abc",
        evals_path="evals.json",
        out_dir="out",
        workdir="/app",
        base_image="python:3.12-slim",
        task_version="1.0",
        cases=[HarborCase(case_id="x", task_names=["case-x__with", "case-x__without"], arm="with")],
        tasks={"case-x__with": "x", "case-x__without": "x"},
        task_arms={"case-x__with": "with", "case-x__without": "without"},
    )

    assert _arm_task_names(manifest, True) == ["case-x__with"]
    assert _arm_task_names(manifest, False) == ["case-x__without"]


def test_run_harbor_requires_an_explicit_model(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("agent_skill_eval.harbor.run.require_harbor", lambda: "0.26.0")

    with pytest.raises(ValueError, match="explicit model"):
        run_harbor(
            HarborRunOptions(
                evals_path=EVALS,
                skill_path=SKILL,
                workspace=tmp_path / "workspace",
                agents=[AgentType.OPENCODE],
                agent_models={},
            )
        )


def test_run_harbor_exports_runs_and_imports(tmp_path: Path, monkeypatch):
    template = _write_template(tmp_path)
    monkeypatch.setenv("FAKE_JOB_TEMPLATE", str(template))
    monkeypatch.setattr("agent_skill_eval.harbor.run.require_harbor", lambda: "0.26.0")

    fake = tmp_path / "fake-harbor"
    fake.write_text(f"#!{sys.executable}\n" + FAKE_HARBOR)
    fake.chmod(0o755)

    iteration_dir = run_harbor(
        HarborRunOptions(
            evals_path=EVALS,
            skill_path=SKILL,
            workspace=tmp_path / "workspace",
            agents=[AgentType.OPENCODE],
            agent_models={AgentType.OPENCODE: "openrouter/deepseek/deepseek-v4.1-flash"},
            runs=1,
            concurrency=1,
            harbor_bin=str(fake),
        )
    )

    summary = json.loads((iteration_dir / "summary.json").read_text())
    assert summary["runs_completed"] == 2
    assert summary["pass_rates"]["opencode_with_skill"] == 1.0
    assert summary["pass_rates"]["opencode_without_skill"] == 1.0
    assert summary["trigger_rate"]["opencode_with_skill"]["rate"] == 1.0
    assert summary["trigger_rate"]["opencode_without_skill"]["unknown"] == 1
    assert summary["runs_failed_grading"] == 0

    benchmark = json.loads((iteration_dir / "benchmark.json").read_text())
    assert benchmark["deltas"]["opencode"]["pass_rate"] == 0.0

    run_harbor(
        HarborRunOptions(
            evals_path=EVALS,
            skill_path=SKILL,
            workspace=tmp_path / "workspace",
            agents=[AgentType.OPENCODE],
            agent_models={AgentType.OPENCODE: "openrouter/deepseek/deepseek-v4.1-flash"},
            runs=1,
            concurrency=1,
            harbor_bin=str(fake),
        )
    )
    second = json.loads((iteration_dir / "summary.json").read_text())
    assert second["run_id"] != summary["run_id"]
    jobs_root = iteration_dir / "harbor" / "jobs"
    assert (jobs_root / summary["run_id"]).is_dir()
    assert (jobs_root / second["run_id"]).is_dir()


def _fake_harbor(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("FAKE_JOB_TEMPLATE", str(_write_template(tmp_path)))
    monkeypatch.setattr("agent_skill_eval.harbor.run.require_harbor", lambda: "0.26.0")
    fake = tmp_path / "fake-harbor"
    fake.write_text(f"#!{sys.executable}\n" + FAKE_HARBOR)
    fake.chmod(0o755)
    return fake


def _options(tmp_path: Path, fake: Path, **kwargs) -> HarborRunOptions:
    return HarborRunOptions(
        evals_path=EVALS,
        skill_path=SKILL,
        workspace=tmp_path / "workspace",
        agents=[AgentType.OPENCODE],
        agent_models={AgentType.OPENCODE: "openrouter/deepseek/deepseek-v4.1-flash"},
        harbor_bin=str(fake),
        **kwargs,
    )


def test_missing_trials_are_reported_as_errors(tmp_path: Path, monkeypatch):
    fake = _fake_harbor(tmp_path, monkeypatch)
    # The fake writes one trial per job, but two attempts were requested.
    iteration_dir = run_harbor(_options(tmp_path, fake, runs=2))

    summary = json.loads((iteration_dir / "summary.json").read_text())
    missing = [item for item in summary["errors"] if "no trial" in item["error"]]
    runs = {item["run"] for item in missing}
    assert {"content-contains-opencode-with-r2", "content-contains-opencode-without-r2"} <= runs
    # The one trial the fake wrote is not reported as missing.
    assert "content-contains-opencode-with-r1" not in runs


def test_a_failed_arm_keeps_the_other_arm_and_fails_the_run(tmp_path: Path, monkeypatch):
    fake = _fake_harbor(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_FAIL_WITHOUT", "1")

    with pytest.raises(RuntimeError, match="without_skill"):
        run_harbor(_options(tmp_path, fake))

    summary = json.loads(
        (tmp_path / "workspace" / "format-json-workspace" / "iteration-1" / "summary.json").read_text()
    )
    assert summary["pass_rates"]["opencode_with_skill"] == 1.0


def test_pre_run_hook_failure_aborts_before_harbor(tmp_path: Path, monkeypatch):
    from agent_skill_eval.hooks import HookError

    fake = _fake_harbor(tmp_path, monkeypatch)
    with pytest.raises(HookError):
        run_harbor(_options(tmp_path, fake, pre_run_commands=["exit 3"]))
    assert not list((tmp_path / "workspace").rglob("jobs/*/*"))


def test_post_run_hook_is_recorded_in_the_summary(tmp_path: Path, monkeypatch):
    fake = _fake_harbor(tmp_path, monkeypatch)
    iteration_dir = run_harbor(_options(tmp_path, fake, post_run_commands=["echo done"]))
    summary = json.loads((iteration_dir / "summary.json").read_text())
    assert summary["hooks"]["post_run"][0]["exit_code"] == 0


def _config_for(tmp_path: Path, api_key_name: str | None) -> dict:
    return _job_config(
        job_name="demo",
        jobs_dir=tmp_path / "jobs",
        tasks_dir=tmp_path / "tasks",
        task_names=["case-a"],
        agent=AgentType.OPENCODE,
        model="opencode-go/deepseek-v4.1-flash",
        with_skill=False,
        skill_path=SKILL,
        runs=1,
        concurrency=1,
        base_url=None,
        api_key_name=api_key_name,
        reasoning_effort=None,
        timeout=None,
        retries=None,
    )


def test_job_config_passes_api_key_name_to_agent_entry(tmp_path):
    config = _config_for(tmp_path, "OPENCODE_API_KEY")
    assert config["agents"][0]["model_api_key_name"] == "OPENCODE_API_KEY"


def test_job_config_omits_api_key_name_by_default(tmp_path):
    config = _config_for(tmp_path, None)
    assert "model_api_key_name" not in config["agents"][0]
