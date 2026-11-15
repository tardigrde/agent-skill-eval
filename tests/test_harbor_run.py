from __future__ import annotations

import json
import sys
from pathlib import Path

from agent_skill_eval.harbor.run import HarborRunOptions, run_harbor
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
