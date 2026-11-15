from __future__ import annotations

import os
import subprocess
from pathlib import Path

from agent_skill_eval.harbor.baseline import build_baseline
from agent_skill_eval.harbor.verifier import grade


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
    subprocess.run(["git", "-C", str(workspace), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(workspace), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(workspace), "add", "-A"], check=True)
    env = {
        **os.environ,
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


def test_deferred_assertions_are_skipped_in_the_verifier(tmp_path: Path):
    workspace, baseline = _make_workspace(tmp_path)
    case = _case(workspace, ["The output contains `formatted-json-ok`"], deferred=[0])

    results, _payload = grade(case, baseline)

    assert results[0].skipped is True
    assert results[0].method == "deferred"
