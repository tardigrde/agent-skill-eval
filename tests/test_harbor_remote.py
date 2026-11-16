from __future__ import annotations

import json
from pathlib import Path

from agent_skill_eval.harbor import remote as remote_module
from agent_skill_eval.harbor.importer import import_job
from agent_skill_eval.harbor.remote import (
    attribute_remote_changes,
    check_contract,
    check_pr_created,
    check_pushed,
    read_workspace_head,
    snapshot_remote,
)
from agent_skill_eval.models import GitStateSnapshot
from tests.test_harbor_import import _manifest, _write_job

SHA = "a" * 40
PUSHED = "The branch was pushed to the remote"
PR = "A pull request was created"


def _pre() -> GitStateSnapshot:
    return GitStateSnapshot(remote_branches=["main"], remote_branch_heads={"main": "1" * 40}, open_prs=[])


def _post(pr_state: str = "OPEN") -> GitStateSnapshot:
    return GitStateSnapshot(
        remote_branches=["feature-x", "main", "other-run"],
        remote_branch_heads={"main": "1" * 40, "feature-x": SHA, "other-run": "2" * 40},
        open_prs=[
            {"number": 7, "headRefName": "feature-x", "state": pr_state},
            {"number": 8, "headRefName": "other-run", "state": "OPEN"},
        ],
    )


def test_changes_are_attributed_by_branch_name():
    changes = attribute_remote_changes(_pre(), _post(), "git push origin feature-x")
    assert changes["branches"] == ["feature-x"]
    assert [pr["number"] for pr in changes["prs"]] == [7]
    assert changes["unattributed_branches"] == ["other-run"]


def test_pushed_and_pr_checks_pass_for_attributed_changes():
    changes = attribute_remote_changes(_pre(), _post(), "git push origin feature-x")
    assert check_pushed(PUSHED, changes, True, SHA).passed
    assert check_pr_created(PR, changes, True).passed


def test_pushed_fails_when_remote_head_differs_from_local_head():
    changes = attribute_remote_changes(_pre(), _post(), "git push origin feature-x")
    assert not check_pushed(PUSHED, changes, True, "3" * 40).passed


def test_negative_control_fails_on_a_push_and_passes_without_one():
    changes = attribute_remote_changes(_pre(), _post(), "git push origin feature-x")
    assert not check_pushed(PUSHED, changes, False, None).passed
    assert not check_pr_created(PR, changes, False).passed
    empty = attribute_remote_changes(_pre(), _pre(), "nothing")
    assert check_pushed(PUSHED, empty, False, None).passed
    assert check_pr_created(PR, empty, False).passed


def test_closed_pr_does_not_count_as_created():
    changes = attribute_remote_changes(_pre(), _post("CLOSED"), "git push origin feature-x")
    assert not check_pr_created(PR, changes, True).passed


def test_contract_flags_remote_branches_and_reviews():
    changes = attribute_remote_changes(_pre(), _post(), "git push origin feature-x")
    result = check_contract({"allow_new_remote_branches": False, "allow_new_review_requests": False}, changes)
    assert result is not None and not result.passed
    assert "feature-x" in result.evidence and "7" in result.evidence
    assert check_contract({}, changes) is None


def test_read_workspace_head_reads_files_without_git(tmp_path: Path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/feature-x\n")
    (git / "refs" / "heads" / "feature-x").write_text(SHA + "\n")
    assert read_workspace_head(tmp_path) == SHA
    (git / "refs" / "heads" / "feature-x").unlink()
    (git / "packed-refs").write_text(f"# pack\n{SHA} refs/heads/feature-x\n")
    assert read_workspace_head(tmp_path) == SHA


def test_read_workspace_head_rejects_traversal_and_symlinks(tmp_path: Path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/../../evil\n")
    assert read_workspace_head(tmp_path) is None
    link = tmp_path / "linked"
    link.mkdir()
    (link / ".git").symlink_to(git, target_is_directory=True)
    assert read_workspace_head(link) is None


def _import(tmp_path: Path, remote, **job_args):
    job_dir = _write_job(tmp_path, assertions=[PUSHED, PR], deferred=[0, 1], **job_args)
    details_path = next(job_dir.glob("*/verifier/reward-details.json"))
    details = json.loads(details_path.read_text())
    details["assertions"] = []
    details["case"]["side_effect_contract"] = {"allow_new_remote_branches": False}
    details_path.write_text(json.dumps(details))
    results = import_job(
        job_dir,
        manifest=_manifest(tmp_path),
        iteration_dir=tmp_path / "iteration",
        iteration=1,
        runs=1,
        remote=remote,
    )
    return results["hello-opencode-with-r1"]


def test_import_grades_remote_assertions_on_the_host(tmp_path: Path):
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {
                "step_id": 2,
                "source": "agent",
                "message": "done",
                "tool_calls": [
                    {
                        "tool_call_id": "c1",
                        "function_name": "bash",
                        "arguments": {"command": "git push origin feature-x"},
                    }
                ],
            }
        ],
    }
    run = _import(tmp_path, (_pre(), _post()), trajectory=trajectory)
    results = {item["text"]: item for item in run["grading"]["assertion_results"]}
    assert results[PUSHED]["passed"] and results[PUSHED]["method"] == "host-remote"
    assert results[PR]["passed"]
    contract = results["The run stayed inside its remote side-effect contract"]
    assert not contract["passed"]
    assert run["remote_changes"]["branches"] == ["feature-x"]


def test_import_fails_closed_without_source_repo(tmp_path: Path):
    run = _import(tmp_path, None)
    results = {item["text"]: item for item in run["grading"]["assertion_results"]}
    assert not results[PUSHED]["passed"]
    assert "--source-repo" in results[PUSHED]["evidence"]
    assert not results["The run stayed inside its remote side-effect contract"]["passed"]


def test_snapshot_uses_gh_api_for_github_repos(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(args, cwd):
        calls.append(args)
        return f"refs/heads/main {'1' * 40}\nrefs/heads/feature/x {SHA}\n"

    monkeypatch.setattr(remote_module, "_run", fake_run)
    monkeypatch.setattr(remote_module, "list_review_requests", lambda repo: [])
    state = snapshot_remote("https://github.com/owner/repo.git")
    assert calls[0][:2] == ["gh", "api"] and "repos/owner/repo/git/matching-refs/heads" in calls[0]
    assert state.remote_branch_heads == {"main": "1" * 40, "feature/x": SHA}


def test_snapshot_uses_ls_remote_for_other_hosts(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(args, cwd):
        calls.append(args)
        return f"{SHA}\trefs/heads/main\n"

    monkeypatch.setattr(remote_module, "_run", fake_run)
    monkeypatch.setattr(remote_module, "list_review_requests", lambda repo: [])
    state = snapshot_remote("https://example.com/owner/repo.git")
    assert calls[0][:2] == ["git", "ls-remote"]
    assert state.remote_branches == ["main"]
