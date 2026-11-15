from __future__ import annotations

import json
from pathlib import Path

from agent_skill_eval.harbor.export import load_manifest
from agent_skill_eval.harbor.importer import import_job
from agent_skill_eval.harbor.models import HarborCase, HarborExportManifest

FIXTURES = Path(__file__).parent / "fixtures" / "harbor"
RECORDED_JOBS = FIXTURES / "jobs"
RECORDED_EXPORT = FIXTURES / "export"


def _manifest(tmp_path: Path) -> HarborExportManifest:
    return HarborExportManifest(
        suite="demo",
        skill_name="demo",
        skill_frontmatter_name="demo",
        skill_hash="abc",
        evals_path=str(tmp_path / "evals.json"),
        out_dir=str(tmp_path / "export"),
        workdir="/app",
        base_image="python:3.12-slim",
        task_version="1.0",
        cases=[HarborCase(case_id="hello", task_names=["case-hello"], arm="both")],
        tasks={"case-hello": "hello"},
        task_arms={"case-hello": "both"},
    )


def _write_job(
    tmp_path: Path,
    *,
    output_tokens: int | None = 10,
    exception: dict | None = None,
    assertions: list | None = None,
    deferred: list[int] | None = None,
    trajectory: dict | None = None,
) -> Path:
    job_dir = tmp_path / "job"
    trial = job_dir / "case-hello__abc123"
    for path in (trial / "verifier", trial / "agent", trial / "artifacts" / "app"):
        path.mkdir(parents=True, exist_ok=True)
    (trial / "artifacts" / "app" / "hello.txt").write_text("Hello, world!\n")
    (trial / "agent" / "opencode-v2.txt").write_text('{"type":"text","text":"Hello, world!"}\n')

    assertions = assertions or [
        {"type": "file_exists", "path": "hello.txt"},
        "The output contains `Hello`",
    ]
    deferred = deferred if deferred is not None else [1]
    case = {
        "case_id": "hello",
        "task_name": "case-hello",
        "workdir": "/app",
        "assertions": assertions,
        "deferred": deferred,
        "expected_output": "Hello",
        "should_trigger": True,
        "force_skill_invocation": False,
        "side_effect_contract": None,
        "side_effect_level": "local-only",
        "skill_name": "demo",
        "skill_frontmatter_name": "demo",
    }
    details = {
        "case_id": "hello",
        "task_name": "case-hello",
        "workdir": "/app",
        "case": case,
        "assertions": [
            {
                "index": 0,
                "text": "file_exists (path='hello.txt')",
                "passed": True,
                "skipped": False,
                "method": "structured:file_exists",
                "evidence": "hello.txt exists",
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
    (trial / "verifier" / "reward-details.json").write_text(json.dumps(details))
    (trial / "verifier" / "reward.json").write_text(json.dumps({"correctness": 1.0, "a0": 1.0}))

    if trajectory is None:
        trajectory = {
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
    (trial / "agent" / "trajectory.json").write_text(json.dumps(trajectory))

    result = {
        "task_name": "skill-eval/case-hello",
        "trial_name": "case-hello__abc123",
        "config": {
            "task": {"path": str(tmp_path / "export" / "tasks" / "case-hello")},
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
            "n_output_tokens": output_tokens,
            "cost_usd": 0.001,
        },
        "verifier_result": {"rewards": {"correctness": 1.0}},
        "reward_valid": True,
        "exception_info": exception,
        "agent_execution": {
            "started_at": "2026-11-15T00:00:00+00:00",
            "finished_at": "2026-11-15T00:00:10+00:00",
        },
    }
    (trial / "result.json").write_text(json.dumps(result))
    (job_dir / "result.json").write_text(json.dumps({"schema_version": 1}))
    return job_dir


def test_import_merges_verifier_and_deferred_results(tmp_path: Path):
    job_dir = _write_job(tmp_path)
    results = import_job(
        job_dir,
        manifest=_manifest(tmp_path),
        iteration_dir=tmp_path / "iteration",
        iteration=1,
        runs=1,
    )

    run = results["hello-opencode-with-r1"]
    assert run["grading"]["summary"]["passed"] == 2
    assert run["grading"]["summary"]["total"] == 2
    assert run["skill_invoked"] is True
    assert run["timing"]["total_tokens"] == 110
    assert run["timing"]["non_cached_input_tokens"] == 50
    assert (tmp_path / "iteration" / "eval-hello" / "opencode" / "with_skill" / "grading.json").is_file()


def test_import_treats_zero_output_tokens_as_an_error(tmp_path: Path):
    job_dir = _write_job(tmp_path, output_tokens=0)
    results = import_job(
        job_dir,
        manifest=_manifest(tmp_path),
        iteration_dir=tmp_path / "iteration",
        iteration=1,
        runs=1,
    )

    assert "error" in results["hello-opencode-with-r1"]
    assert "0 output tokens" in results["hello-opencode-with-r1"]["error"]


def test_import_does_not_treat_unknown_output_tokens_as_zero(tmp_path: Path):
    job_dir = _write_job(tmp_path, output_tokens=None)
    results = import_job(
        job_dir,
        manifest=_manifest(tmp_path),
        iteration_dir=tmp_path / "iteration",
        iteration=1,
        runs=1,
    )

    assert "grading" in results["hello-opencode-with-r1"]


def test_import_treats_agent_side_exceptions_as_errors(tmp_path: Path):
    job_dir = _write_job(
        tmp_path,
        exception={"exception_type": "NonZeroAgentExitCodeError", "exception_message": "exit 1"},
    )
    results = import_job(
        job_dir,
        manifest=_manifest(tmp_path),
        iteration_dir=tmp_path / "iteration",
        iteration=1,
        runs=1,
    )

    assert "error" in results["hello-opencode-with-r1"]
    assert "NonZeroAgentExitCodeError" in results["hello-opencode-with-r1"]["error"]


def test_recorded_job_directories_import_without_docker(tmp_path: Path):
    manifest = load_manifest(RECORDED_EXPORT)
    # The manifest is portable; resolve its export directory relative to the fixture.
    manifest.out_dir = str(RECORDED_EXPORT)
    results: dict[str, dict] = {}
    for job in ("fixture-with", "fixture-without"):
        results.update(
            import_job(
                RECORDED_JOBS / job,
                manifest=manifest,
                iteration_dir=tmp_path / "iteration",
                iteration=1,
                runs=1,
            )
        )

    with_skill = results["content-contains-opencode-with-r1"]
    without_skill = results["content-contains-opencode-without-r1"]
    assert with_skill["grading"]["summary"]["passed"] == 1
    assert with_skill["skill_invoked"] is True
    assert without_skill["grading"]["summary"]["passed"] == 1
    assert without_skill["skill_invoked"] is None
    assert with_skill["timing"]["output_tokens"] == 122
    assert with_skill["timing"]["cost_usd"] is not None
    assert with_skill["tags"] == ["correctness"]
    assert with_skill["should_trigger"] is True
    assert isinstance(with_skill["out_of_workspace"], list)


def test_missing_reward_details_is_an_error(tmp_path: Path):
    job_dir = _write_job(tmp_path)
    trial = next(path for path in job_dir.iterdir() if path.is_dir())
    (trial / "verifier" / "reward-details.json").unlink()

    results = import_job(
        job_dir,
        manifest=_manifest(tmp_path),
        iteration_dir=tmp_path / "iteration",
        iteration=1,
        runs=1,
    )

    assert "error" in results["hello-opencode-with-r1"]
    assert "missing" in results["hello-opencode-with-r1"]["error"]


def test_partial_reward_details_is_an_error(tmp_path: Path):
    job_dir = _write_job(tmp_path)
    trial = next(path for path in job_dir.iterdir() if path.is_dir())
    details_path = trial / "verifier" / "reward-details.json"
    details = json.loads(details_path.read_text())
    details["assertions"] = [entry for entry in details["assertions"] if entry["index"] != 0]
    details_path.write_text(json.dumps(details))

    results = import_job(
        job_dir,
        manifest=_manifest(tmp_path),
        iteration_dir=tmp_path / "iteration",
        iteration=1,
        runs=1,
    )

    assert "error" in results["hello-opencode-with-r1"]
    assert "did not report" in results["hello-opencode-with-r1"]["error"]


def test_deferred_results_keep_their_assertion_indices(tmp_path: Path):
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {
                "step_id": 2,
                "source": "agent",
                "message": "running tests",
                "tool_calls": [
                    {
                        "tool_call_id": "c1",
                        "function_name": "shell",
                        "arguments": {"command": "pytest -q"},
                    }
                ],
            }
        ],
    }
    job_dir = _write_job(
        tmp_path,
        assertions=["The summary is friendly", {"type": "command_ran", "command": "pytest"}],
        deferred=[0, 1],
        trajectory=trajectory,
    )
    results = import_job(
        job_dir,
        manifest=_manifest(tmp_path),
        iteration_dir=tmp_path / "iteration",
        iteration=1,
        runs=1,
    )

    assertions = results["hello-opencode-with-r1"]["grading"]["assertion_results"]
    assert assertions[0]["text"] == "The summary is friendly"
    assert assertions[0]["skipped"] is True
    assert assertions[1]["passed"] is True
    assert "pytest" in assertions[1]["text"]


def test_import_records_run_id_and_workspace_path(tmp_path: Path):
    job_dir = _write_job(tmp_path)
    import_job(
        job_dir,
        manifest=_manifest(tmp_path),
        iteration_dir=tmp_path / "iteration",
        iteration=1,
        runs=1,
        run_id="abc123",
    )

    config_dir = tmp_path / "iteration" / "eval-hello" / "opencode" / "with_skill"
    meta = json.loads((config_dir / "run_meta.json").read_text())
    assert meta["run_id"] == "abc123"
    assert meta["workspace_path"].endswith("artifacts/app")
    assert (config_dir / "outputs" / "output.txt").is_file()


def test_import_grades_command_ran_from_atif(tmp_path: Path):
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {
                "step_id": 2,
                "source": "agent",
                "message": "running tests",
                "tool_calls": [
                    {
                        "tool_call_id": "c1",
                        "function_name": "shell",
                        "arguments": {"command": "pytest -q"},
                    }
                ],
            }
        ],
    }
    job_dir = _write_job(
        tmp_path,
        assertions=[{"type": "command_ran", "command": "pytest"}],
        deferred=[0],
        trajectory=trajectory,
    )
    results = import_job(
        job_dir,
        manifest=_manifest(tmp_path),
        iteration_dir=tmp_path / "iteration",
        iteration=1,
        runs=1,
    )

    run = results["hello-opencode-with-r1"]
    assert run["grading"]["summary"]["passed"] == 1
    assert run["skill_invoked"] is False
