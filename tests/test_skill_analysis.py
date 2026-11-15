from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_skill_eval.analysis import (
    build_baseline,
    compare_to_baseline,
    percentile,
    render_comparison_markdown,
    safe_eval_dir,
    wilson_interval,
)
from agent_skill_eval.cli import app
from agent_skill_eval.summary import _out_of_workspace, _tag_pass_rates, _trigger_rates


def test_safe_eval_dir_rejects_path_traversal_and_collisions():
    assert safe_eval_dir("1") == "eval-1"
    assert safe_eval_dir("../../etc/passwd").startswith("eval-etc-passwd-")
    assert safe_eval_dir("a b").startswith("eval-a-b-")
    assert safe_eval_dir("a/b") != safe_eval_dir("a?b")


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 0.9) == pytest.approx(3.7)
    assert percentile([], 0.5) == 0.0


def test_wilson_interval_bounds():
    low, high = wilson_interval(5, 10)
    assert 0.0 <= low < 0.5 < high <= 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def _summary() -> dict:
    return {
        "skill_name": "demo",
        "skill_hash": "hash",
        "run_id": "run-1",
        "iteration": 1,
        "agents": ["opencode"],
        "stability": {
            "groups": [
                {"eval_id": "1", "agent": "opencode", "with_skill": True, "runs": 2, "full_passes": 2},
                {"eval_id": "2", "agent": "opencode", "with_skill": True, "runs": 2, "full_passes": 1},
                {"eval_id": "1", "agent": "opencode", "with_skill": False, "runs": 2, "full_passes": 1},
                {"eval_id": "2", "agent": "opencode", "with_skill": False, "runs": 2, "full_passes": 1},
            ]
        },
    }


def _benchmark() -> dict:
    return {
        "run_summary": {
            "opencode_with_skill": {
                "pass_rate": {"mean": 0.75, "n": 4},
                "tokens": {"median": 100, "p90": 120},
                "time_seconds": {"median": 1.0, "p90": 2.0},
                "cost_usd": {"median": 0.1, "p90": 0.2},
                "pass_hat_k": 0.5,
                "pass_hat_k_wilson_low": 0.1,
                "pass_hat_k_wilson_high": 0.9,
                "k": 2,
            },
            "opencode_without_skill": {
                "pass_rate": {"mean": 0.5, "n": 4},
                "tokens": {"median": 80, "p90": 90},
                "time_seconds": {"median": 0.8, "p90": 1.0},
                "cost_usd": {"median": 0.08, "p90": 0.1},
                "pass_hat_k": 0.0,
                "pass_hat_k_wilson_low": 0.0,
                "pass_hat_k_wilson_high": 0.5,
                "k": 2,
            },
        }
    }


def test_baseline_round_trip_and_comparison(tmp_path):
    baseline = build_baseline(_summary(), _benchmark())
    assert baseline["configs"]["opencode_with_skill"]["pass_hat_k"] == 0.5
    assert baseline["cases"]["opencode_with_skill"]["1"] == 1.0

    comparison = compare_to_baseline(_summary(), _benchmark(), baseline)
    assert comparison["configs"]["opencode_with_skill"]["pass_rate"]["delta"] == pytest.approx(0.0)
    assert comparison["case_delta"]["opencode_with_skill"]["mean"] == pytest.approx(0.0)
    assert "Baseline comparison" in render_comparison_markdown(comparison)


def test_trigger_rates_split_positive_and_negative_controls():
    runs = [
        {"agent": "opencode", "with_skill": True, "should_trigger": True, "skill_invoked": True},
        {"agent": "opencode", "with_skill": True, "should_trigger": True, "skill_invoked": None},
        {"agent": "opencode", "with_skill": True, "should_trigger": False, "skill_invoked": False},
    ]
    rates = _trigger_rates(runs)["opencode_with_skill"]

    assert rates["positive"]["rate"] == 1.0
    assert rates["positive"]["unknown"] == 1
    assert rates["negative"]["rate"] == 0.0
    assert rates["unknown"] == 1


def test_forced_cases_are_excluded_from_trigger_rates():
    runs = [
        {
            "agent": "opencode",
            "with_skill": True,
            "should_trigger": True,
            "force_skill_invocation": True,
            "skill_invoked": True,
        }
    ]
    assert _trigger_rates(runs) == {}


def test_tag_pass_rates():
    runs = [
        {
            "agent": "opencode",
            "with_skill": True,
            "tags": ["correctness", "skill_effect"],
            "grading": {"summary": {"pass_rate": 1.0}},
        },
        {
            "agent": "opencode",
            "with_skill": True,
            "tags": ["correctness"],
            "grading": {"summary": {"pass_rate": 0.5}},
        },
    ]
    rates = _tag_pass_rates(runs)["opencode_with_skill"]

    assert rates["correctness"] == 0.75
    assert rates["skill_effect"] == 1.0


FIXTURES = Path(__file__).parent / "fixtures"


def test_run_writes_and_compares_a_baseline(tmp_path):
    runner = CliRunner()
    workspace = tmp_path / "workspace"
    baseline = tmp_path / "baseline.json"
    common = [
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
        "--eval-id",
        "content-contains",
    ]

    first = runner.invoke(app, [*common, "--write-baseline", str(baseline)])
    assert first.exit_code == 0, first.output
    assert baseline.is_file()
    assert "configs" in baseline.read_text()

    second = runner.invoke(app, [*common, "--iteration", "2", "--compare", str(baseline)])
    assert second.exit_code == 0, second.output
    comparison = workspace / "format-json-workspace" / "iteration-2" / "comparison.md"
    assert comparison.is_file()
    assert "Baseline comparison" in comparison.read_text()


def test_out_of_workspace_examples_are_bounded():
    runs = [
        {"agent": "opencode", "with_skill": True, "out_of_workspace": ["read:/etc/passwd"]},
        {"agent": "opencode", "with_skill": True, "out_of_workspace": ["read:/etc/passwd", "shell:/tmp/x"]},
    ]
    entry = _out_of_workspace(runs)["opencode_with_skill"]

    assert entry["runs_with_access"] == 2
    assert entry["examples"] == ["read:/etc/passwd", "shell:/tmp/x"]
