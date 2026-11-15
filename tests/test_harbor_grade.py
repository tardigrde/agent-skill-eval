from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_skill_eval.analysis import safe_eval_dir
from agent_skill_eval.cli import app


def test_grade_uses_harbor_metadata_and_saved_output(tmp_path: Path):
    iteration = tmp_path / "iteration-1"
    config_dir = iteration / safe_eval_dir("a/b") / "opencode" / "with_skill"
    outputs = config_dir / "outputs"
    outputs.mkdir(parents=True)
    workspace = tmp_path / "artifact"
    workspace.mkdir()

    (iteration / "evals_meta.json").write_text(
        json.dumps(
            {
                "skill_name": "demo",
                "skill_hash": "hash",
                "evals": [
                    {
                        "id": "a/b",
                        "prompt": "Run the tests.",
                        "expected_output": "Tests pass.",
                        "assertions": [{"type": "command_ran", "command": "pytest"}],
                        "should_trigger": True,
                    }
                ],
            }
        )
    )
    (config_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "eval_id": "a/b",
                "agent": "opencode",
                "with_skill": True,
                "run_id": "abc123",
                "workspace_path": str(workspace),
            }
        )
    )
    (outputs / "output.txt").write_text("pytest ran")
    (outputs / "stdout.log").write_text(json.dumps({"type": "command_execution", "command": "pytest -q"}) + "\n")
    (config_dir / "grading.json").write_text(json.dumps({"assertion_results": [], "summary": {}}))
    (config_dir / "timing.json").write_text(json.dumps({"total_tokens": 1}))

    result = CliRunner().invoke(
        app,
        ["grade", "--workspace", str(iteration), "--grader-model", ""],
    )

    assert result.exit_code == 0, result.output
    assert "No evals_meta.json" not in result.output
    grading = json.loads((config_dir / "grading.json").read_text())
    assert grading["summary"]["passed"] == 1
    assert grading["summary"]["total"] == 1
