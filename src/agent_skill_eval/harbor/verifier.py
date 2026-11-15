"""Deterministic verifier entry point for exported Harbor tasks.

Runs inside the task's verifier container. It grades workspace assertions only.
Output-derived assertions are marked deferred and graded by the importer, which
has the agent transcript.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from agent_skill_eval.git_state import capture_git_state, safe_git_env, state_diff
from agent_skill_eval.graders import (
    DeterministicGrader,
    assertion_text,
    summarize_assertion_results,
)
from agent_skill_eval.harbor.baseline import (
    baseline_commit,
    baseline_commit_subject,
    baseline_tree,
    baseline_violations,
    is_ancestor,
    is_git_repo,
)
from agent_skill_eval.harbor.models import HarborBaseline
from agent_skill_eval.models import AssertionResult, GitStateSnapshot, StructuredAssertion

_GIT_PROSE = {"git-branch", "git-commit"}


def _tests_dir() -> Path:
    override = os.environ.get("HARBOR_TESTS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]


def _structured(assertion: Any) -> StructuredAssertion | None:
    if isinstance(assertion, dict):
        try:
            return StructuredAssertion(**assertion)
        except Exception:
            return None
    return None


def _result(
    text: str,
    passed: bool,
    evidence: str,
    method: str,
    *,
    skipped: bool = False,
) -> AssertionResult:
    return AssertionResult(
        text=text,
        passed=passed,
        evidence=evidence,
        method=method,
        skipped=skipped,
    )


def _file_unchanged(
    assertion: StructuredAssertion,
    workspace: Path,
    baseline: HarborBaseline,
) -> AssertionResult:
    text = assertion_text(assertion)
    if not assertion.path:
        return _result(text, False, "file_unchanged requires a path", "structured:file_unchanged")
    violations = baseline_violations(workspace, baseline, only=assertion.path)
    if violations:
        return _result(
            text,
            False,
            "Baseline difference: " + "; ".join(violations),
            "structured:file_unchanged",
        )
    return _result(text, True, f"{assertion.path} matches the baseline", "structured:file_unchanged")


def _git_clean(
    assertion: StructuredAssertion,
    workspace: Path,
    baseline: HarborBaseline,
    grader: DeterministicGrader,
) -> AssertionResult:
    text = assertion_text(assertion)
    if not is_git_repo(workspace):
        return _result(
            text,
            False,
            "Git history is missing from the worktree",
            "structured:git_clean",
        )
    violations = baseline_violations(workspace, baseline)
    if violations:
        return _result(
            text,
            False,
            "Worktree differs from the baseline: " + "; ".join(violations),
            "structured:git_clean",
        )
    native = grader.check_one(assertion, _tests_dir(), "", workspace)
    if not native.passed:
        return native
    return _result(text, True, "Worktree matches the baseline", "structured:git_clean")


def _git_prose(
    assertion: str,
    workspace: Path,
    grader: DeterministicGrader,
    pre: Optional[GitStateSnapshot],
    post: Optional[GitStateSnapshot],
    should_trigger: bool,
    baseline_trusted: bool,
) -> AssertionResult:
    text = assertion_text(assertion)
    if not baseline_trusted:
        return _result(
            text,
            False,
            "Baseline commit tree does not match tests/baseline.json",
            "git",
        )
    if pre is None or post is None:
        return _result(
            text,
            False,
            "Git history is unavailable in the worktree",
            "git",
        )
    if pre.head_sha and not is_ancestor(workspace, pre.head_sha):
        return _result(
            text,
            False,
            "Baseline commit is missing from the worktree history",
            "git",
        )
    diff = state_diff(pre, post)
    grader.pre_state = pre
    grader.post_state = post
    return grader._check_assertion(assertion, _tests_dir(), "", workspace, should_trigger, diff)


def _runtime_state(
    workspace: Path,
    baseline: HarborBaseline,
) -> tuple[Optional[GitStateSnapshot], Optional[GitStateSnapshot], bool, bool]:
    if not is_git_repo(workspace):
        return None, None, False, True
    sha = baseline_commit(workspace)
    if not sha:
        return None, None, False, True
    trusted = True
    if baseline.commit_sha and sha != baseline.commit_sha:
        trusted = False
    if baseline.tree_hash and baseline_tree(workspace, sha) != baseline.tree_hash:
        trusted = False
    subject = baseline_commit_subject(workspace, sha)
    pre = GitStateSnapshot(
        local_branches=["main"],
        current_branch="main",
        head_sha=sha,
        commit_count=1,
        commits=[subject] if subject else [],
        commit_shas=[sha],
        branch_heads={"main": sha},
    )
    post = capture_git_state(workspace)
    return pre, post, True, trusted


def _side_effect_contract(
    contract: dict[str, Any],
    workspace: Path,
    baseline: HarborBaseline,
    pre: Optional[GitStateSnapshot],
    post: Optional[GitStateSnapshot],
    has_git: bool,
    baseline_trusted: bool,
) -> AssertionResult:
    problems: list[str] = []
    evidence: list[str] = []

    violations = baseline_violations(workspace, baseline)
    if not contract.get("allow_worktree_changes", True):
        if violations:
            problems.append("worktree changes are not allowed")
            evidence.extend(violations)
        # A new symlink is deliberately skipped by the hash walk. Git still
        # reports it as untracked, so use the status snapshot as well.
        if post is not None and post.status_porcelain:
            if "worktree changes are not allowed" not in problems:
                problems.append("worktree changes are not allowed")
            evidence.extend(post.status_porcelain[:20])
    if not baseline_trusted:
        problems.append("baseline commit tree does not match tests/baseline.json")

    if has_git and pre is not None and post is not None:
        diff = state_diff(pre, post)
        if diff.get("new_branches") and not contract.get("allow_new_local_branches", True):
            problems.append("new local branches are not allowed")
            evidence.append(f"new branches: {diff['new_branches']}")
        if diff.get("new_commit_shas") and not contract.get("allow_new_commits", True):
            problems.append("new commits are not allowed")
            evidence.append(f"new commits: {diff['new_commit_shas']}")
    elif not contract.get("allow_new_local_branches", True) or not contract.get("allow_new_commits", True):
        problems.append("git history is unavailable; branch/commit contract cannot be verified")

    if not contract.get("allow_new_remote_branches", True):
        problems.append("remote branch checks are unavailable in the verifier")
    if not contract.get("allow_new_review_requests", True):
        problems.append("review-request checks are unavailable in the verifier")

    return _result(
        "The run stayed inside its side-effect contract",
        not problems,
        "; ".join(problems + evidence) or "No side-effect violations",
        "side-effect-contract",
    )


def _git_probe(workspace: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "log", "--all", "--pretty=format:%H"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
            env=safe_git_env(workspace),
        )
    except OSError as exc:
        return {"error": str(exc)}
    return {
        "returncode": result.returncode,
        "commits": len([line for line in result.stdout.splitlines() if line.strip()]),
        "stderr": (result.stderr or "").strip()[:500],
    }


def grade(case: dict[str, Any], baseline: HarborBaseline) -> tuple[list[AssertionResult], dict[str, Any]]:
    workspace = Path(case["workdir"])
    assertions: list[Any] = case.get("assertions") or []
    deferred = {int(index) for index in case.get("deferred") or []}
    should_trigger = bool(case.get("should_trigger", True))
    pre, post, has_git, baseline_trusted = _runtime_state(workspace, baseline)
    grader = DeterministicGrader(pre_state=pre, post_state=post)
    if pre is not None and post is not None:
        grader.pre_state = pre
        grader.post_state = post

    results: list[AssertionResult] = []
    details: list[dict[str, Any]] = []
    for index, assertion in enumerate(assertions):
        text = assertion_text(assertion)
        if index in deferred:
            result = _result(
                text,
                False,
                "Output-derived assertion; graded by the importer",
                "deferred",
                skipped=True,
            )
        else:
            structured = _structured(assertion)
            if structured is not None and structured.type == "file_unchanged":
                result = _file_unchanged(structured, workspace, baseline)
            elif structured is not None and structured.type == "git_clean":
                result = _git_clean(structured, workspace, baseline, grader)
            elif isinstance(assertion, str) and _GIT_PROSE.intersection(_route_names(assertion)):
                result = _git_prose(
                    assertion,
                    workspace,
                    grader,
                    pre,
                    post,
                    should_trigger,
                    baseline_trusted,
                )
            else:
                result = grader.check_one(
                    assertion,
                    _tests_dir(),
                    "",
                    workspace=workspace,
                    should_trigger=should_trigger,
                )
        results.append(result)
        details.append(
            {
                "index": index,
                "text": result.text,
                "passed": result.passed,
                "skipped": result.skipped,
                "method": result.method,
                "evidence": result.evidence,
                "deferred": index in deferred,
            }
        )

    contract = case.get("side_effect_contract")
    if contract:
        contract_result = _side_effect_contract(
            contract,
            workspace,
            baseline,
            pre,
            post,
            has_git,
            baseline_trusted,
        )
        results.append(contract_result)
        details.append(
            {
                "index": len(assertions),
                "text": contract_result.text,
                "passed": contract_result.passed,
                "skipped": contract_result.skipped,
                "method": contract_result.method,
                "evidence": contract_result.evidence,
                "deferred": False,
            }
        )

    summary = summarize_assertion_results(results)
    payload = {
        "case_id": case.get("case_id"),
        "task_name": case.get("task_name"),
        "workdir": case.get("workdir"),
        "case": case,
        "assertions": details,
        "summary": summary.model_dump(mode="json"),
        "has_git": has_git,
        "baseline_trusted": baseline_trusted,
        "git_probe": _git_probe(workspace),
        "baseline_violations": baseline_violations(workspace, baseline),
    }
    return results, payload


def _route_names(assertion: str) -> set[str]:
    from agent_skill_eval.graders import classify_assertion

    return {classify_assertion(assertion)}


def reward_payload(results: list[AssertionResult]) -> dict[str, float]:
    graded = [result for result in results if not result.skipped]
    passed = sum(1 for result in graded if result.passed)
    # No graded assertion means no evidence. Never hand Harbor a free pass;
    # the importer replaces this value when every check was deferred.
    correctness = passed / len(graded) if graded else 0.0
    # Harbor's viewers and uploaders read the "reward" key. "correctness" is
    # kept as a labelled aggregate next to the per-assertion keys.
    reward: dict[str, float] = {"reward": correctness, "correctness": correctness}
    for index, result in enumerate(results):
        if not result.skipped:
            reward[f"a{index}"] = 1.0 if result.passed else 0.0
    return reward


def write_rewards(
    results: list[AssertionResult],
    payload: dict[str, Any],
    target: Optional[Path] = None,
) -> None:
    target = target or Path("/logs/verifier")
    target.mkdir(parents=True, exist_ok=True)
    reward = reward_payload(results)
    (target / "reward.json").write_text(json.dumps(reward, indent=2) + "\n", encoding="utf-8")
    (target / "reward-details.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    del argv
    tests_dir = _tests_dir()
    case = json.loads((tests_dir / "case.json").read_text(encoding="utf-8"))
    baseline = HarborBaseline.model_validate_json((tests_dir / "baseline.json").read_text(encoding="utf-8"))
    results, payload = grade(case, baseline)
    write_rewards(results, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
