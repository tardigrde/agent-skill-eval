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

from agent_skill_eval.git_state import capture_git_state, state_diff
from agent_skill_eval.graders import (
    DeterministicGrader,
    assertion_text,
    summarize_assertion_results,
)
from agent_skill_eval.harbor.baseline import (
    baseline_commit,
    baseline_commit_subject,
    baseline_violations,
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
    violations = baseline_violations(workspace, baseline)
    if violations:
        return _result(
            text,
            False,
            "Worktree differs from the baseline: " + "; ".join(violations),
            "structured:git_clean",
        )
    if is_git_repo(workspace):
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
) -> AssertionResult:
    text = assertion_text(assertion)
    if pre is None or post is None:
        return _result(
            text,
            False,
            "Git history is unavailable in the verifier; deferred to the importer",
            "git",
            skipped=True,
        )
    if pre.head_sha and pre.head_sha not in post.commit_shas:
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


def _runtime_state(workspace: Path) -> tuple[Optional[GitStateSnapshot], Optional[GitStateSnapshot], bool]:
    if not is_git_repo(workspace):
        return None, None, False
    sha = baseline_commit(workspace)
    if not sha:
        return None, None, False
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
    return pre, post, True


def _ensure_git_safe(workspace: Path) -> None:
    """Allow git to read an artifact restored under a different uid.

    Harbor copies the workdir into the verifier; ownership does not always
    match the verifier user. The verifier is isolated, so trusting the
    restored workdir is safe.
    """
    for directory in (str(workspace), "*"):
        try:
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", directory],
                check=False,
                capture_output=True,
            )
        except OSError:
            return


def _git_probe(workspace: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "log", "--all", "--pretty=format:%H"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
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
    _ensure_git_safe(workspace)
    assertions: list[Any] = case.get("assertions") or []
    deferred = {int(index) for index in case.get("deferred") or []}
    should_trigger = bool(case.get("should_trigger", True))
    pre, post, has_git = _runtime_state(workspace)
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
                result = _git_prose(assertion, workspace, grader, pre, post, should_trigger)
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

    summary = summarize_assertion_results(results)
    payload = {
        "case_id": case.get("case_id"),
        "task_name": case.get("task_name"),
        "workdir": case.get("workdir"),
        "assertions": details,
        "summary": summary.model_dump(mode="json"),
        "has_git": has_git,
        "git_probe": _git_probe(workspace),
        "baseline_violations": baseline_violations(workspace, baseline),
    }
    return results, payload


def _route_names(assertion: str) -> set[str]:
    from agent_skill_eval.graders import classify_assertion

    return {classify_assertion(assertion)}


def write_rewards(results: list[AssertionResult], payload: dict[str, Any]) -> None:
    graded = [result for result in results if not result.skipped]
    passed = sum(1 for result in graded if result.passed)
    correctness = passed / len(graded) if graded else 1.0
    reward: dict[str, float] = {"correctness": correctness}
    for index, result in enumerate(results):
        if not result.skipped:
            reward[f"a{index}"] = 1.0 if result.passed else 0.0

    target = Path("/logs/verifier")
    target.mkdir(parents=True, exist_ok=True)
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
