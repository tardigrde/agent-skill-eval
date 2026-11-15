from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from agent_skill_eval.harbor.baseline import build_baseline
from agent_skill_eval.harbor.verifier import grade, reward_payload, write_rewards
from agent_skill_eval.models import AssertionResult


def _make_workspace(tmp_path: Path) -> tuple[Path, object]:
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "hello.txt").write_text("Hello, world!\n")
    (sources / "keep.txt").write_text("do not touch\n")
    fixture_map = {
        "hello.txt": sources / "hello.txt",
        "keep.txt": sources / "keep.txt",
    }
    baseline = build_baseline(fixture_map, "/app")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for target, source in fixture_map.items():
        (workspace / target).write_bytes(source.read_bytes())

    subprocess.run(["git", "init", "-q", "-b", "main", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "config", "user.email", "eval@agent-skill-eval.local"], check=True)
    subprocess.run(["git", "-C", str(workspace), "config", "user.name", "agent-skill-eval"], check=True)
    subprocess.run(["git", "-C", str(workspace), "add", "-A"], check=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "agent-skill-eval",
        "GIT_AUTHOR_EMAIL": "eval@agent-skill-eval.local",
        "GIT_COMMITTER_NAME": "agent-skill-eval",
        "GIT_COMMITTER_EMAIL": "eval@agent-skill-eval.local",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
    }
    subprocess.run(["git", "-C", str(workspace), "commit", "-qm", "baseline"], check=True, env=env)
    sha = subprocess.check_output(["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True).strip()
    (workspace / ".git" / "eval-baseline").write_text(sha + "\n")
    return workspace, baseline


def _case(workspace: Path, assertions: list, *, should_trigger: bool = True, deferred: list[int] | None = None):
    return {
        "case_id": "case",
        "task_name": "case",
        "workdir": str(workspace),
        "assertions": assertions,
        "deferred": deferred or [],
        "should_trigger": should_trigger,
    }


def test_workspace_assertions_pass_on_an_untouched_tree(tmp_path: Path):
    workspace, baseline = _make_workspace(tmp_path)
    case = _case(
        workspace,
        [
            {"type": "file_exists", "path": "hello.txt"},
            {"type": "file_contains", "path": "hello.txt", "contains": "Hello"},
            {"type": "file_unchanged", "path": "keep.txt"},
            {"type": "git_clean"},
        ],
    )

    results, payload = grade(case, baseline)

    assert [result.passed for result in results] == [True, True, True, True]
    assert payload["has_git"] is True
    assert payload["baseline_violations"] == []


def test_file_unchanged_detects_modification(tmp_path: Path):
    workspace, baseline = _make_workspace(tmp_path)
    (workspace / "keep.txt").write_text("changed\n")
    case = _case(workspace, [{"type": "file_unchanged", "path": "keep.txt"}])

    results, _payload = grade(case, baseline)

    assert results[0].passed is False
    assert "modified: keep.txt" in results[0].evidence


def test_git_clean_detects_a_new_file(tmp_path: Path):
    workspace, baseline = _make_workspace(tmp_path)
    (workspace / "new.txt").write_text("new\n")
    case = _case(workspace, [{"type": "git_clean"}])

    results, _payload = grade(case, baseline)

    assert results[0].passed is False
    assert "created: new.txt" in results[0].evidence


def test_negative_control_passes_when_git_state_is_unchanged(tmp_path: Path):
    workspace, baseline = _make_workspace(tmp_path)
    case = _case(
        workspace,
        ["A new git branch was created", "A git commit was created"],
        should_trigger=False,
    )

    results, _payload = grade(case, baseline)

    assert [result.passed for result in results] == [True, True]


def test_negative_control_fails_when_a_commit_is_created(tmp_path: Path):
    workspace, baseline = _make_workspace(tmp_path)
    (workspace / "new.txt").write_text("new\n")
    subprocess.run(["git", "-C", str(workspace), "add", "new.txt"], check=True)
    subprocess.run(["git", "-C", str(workspace), "commit", "-qm", "agent commit"], check=True)
    case = _case(workspace, ["A git commit was created"], should_trigger=False)

    results, _payload = grade(case, baseline)

    assert results[0].passed is False
    assert "new commit" in results[0].evidence.lower()


def test_side_effect_contract_catches_a_new_symlink(tmp_path: Path):
    workspace, baseline = _make_workspace(tmp_path)
    (workspace / "leak").symlink_to("/etc/hosts")
    case = _case(workspace, [])
    case["side_effect_contract"] = {"allow_worktree_changes": False}

    results, _payload = grade(case, baseline)

    contract = results[-1]
    assert contract.method == "side-effect-contract"
    assert contract.passed is False
    assert "worktree changes" in contract.evidence


def test_missing_git_history_fails_git_checks(tmp_path: Path):
    workspace, baseline = _make_workspace(tmp_path)
    shutil.rmtree(workspace / ".git")
    case = _case(
        workspace,
        [{"type": "git_clean"}, "A git commit was created"],
        should_trigger=False,
    )

    results, payload = grade(case, baseline)

    assert [result.passed for result in results] == [False, False]
    assert "missing" in results[0].evidence.lower()
    assert payload["has_git"] is False


def test_forged_baseline_marker_is_not_trusted(tmp_path: Path):
    workspace, baseline = _make_workspace(tmp_path)
    (workspace / "new.txt").write_text("new\n")
    subprocess.run(["git", "-C", str(workspace), "add", "new.txt"], check=True)
    subprocess.run(["git", "-C", str(workspace), "commit", "-qm", "agent commit"], check=True)
    new_sha = subprocess.check_output(["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True).strip()
    (workspace / ".git" / "eval-baseline").write_text(new_sha + "\n")

    case = _case(workspace, ["A git commit was created"], should_trigger=False)
    results, payload = grade(case, baseline)

    assert results[0].passed is False
    assert "does not match" in results[0].evidence
    assert payload["baseline_trusted"] is False


def test_reward_payload_has_the_primary_reward_key():
    results = [
        AssertionResult(text="a", passed=True, evidence="", method="deterministic"),
        AssertionResult(text="b", passed=False, evidence="", method="deterministic"),
        AssertionResult(text="c", passed=False, evidence="", method="deferred", skipped=True),
    ]

    payload = reward_payload(results)

    assert payload["reward"] == payload["correctness"] == 0.5
    assert payload["a0"] == 1.0
    assert payload["a1"] == 0.0
    assert "a2" not in payload


def test_write_rewards_never_passes_an_empty_verifier(tmp_path: Path):
    write_rewards([], {}, target=tmp_path)
    payload = json.loads((tmp_path / "reward.json").read_text())

    assert payload["reward"] == 0.0
    assert payload["correctness"] == 0.0


def test_deferred_assertions_are_skipped_in_the_verifier(tmp_path: Path):
    workspace, baseline = _make_workspace(tmp_path)
    case = _case(workspace, ["The output contains `formatted-json-ok`"], deferred=[0])

    results, _payload = grade(case, baseline)

    assert results[0].skipped is True
    assert results[0].method == "deferred"


def test_verifier_and_native_grader_agree_on_workspace_assertions(tmp_path: Path):
    """Parity: the Harbor verifier and the native grader give the same verdicts."""
    from agent_skill_eval.graders import grade_assertions

    workspace, baseline = _make_workspace(tmp_path)
    (workspace / "out.json").write_text('{"name": "demo", "items": [1, 2]}')
    assertions = [
        {"type": "file_exists", "path": "out.json"},
        {"type": "file_exists", "path": "missing.txt"},
        {"type": "file_not_exists", "path": "missing.txt"},
        {"type": "file_not_exists", "path": "out.json"},
        {"type": "file_contains", "path": "hello.txt", "text": "Hello"},
        {"type": "file_contains", "path": "hello.txt", "text": "Goodbye"},
        {"type": "file_not_contains", "path": "hello.txt", "text": "Goodbye"},
        {"type": "json_path", "path": "out.json", "json_path": "$.name", "equals": "demo"},
        {"type": "json_path", "path": "out.json", "json_path": "$.name", "equals": "other"},
    ]

    harbor_results, _payload = grade(_case(workspace, assertions), baseline)

    output_dir = tmp_path / "native" / "outputs"
    output_dir.mkdir(parents=True)
    native = grade_assertions(assertions, "", output_dir, "", None, workspace=workspace)

    harbor_verdicts = [(r.passed, r.skipped) for r in harbor_results]
    native_verdicts = [(r.passed, r.skipped) for r in native.assertion_results]
    assert harbor_verdicts == native_verdicts
    assert any(passed for passed, _ in harbor_verdicts)
    assert not all(passed for passed, _ in harbor_verdicts)
