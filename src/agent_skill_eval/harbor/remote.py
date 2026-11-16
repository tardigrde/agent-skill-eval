"""Host-side remote side-effect checks for the Harbor backend.

The verifier runs in a container without a GitHub token, so it cannot see
pushes or pull requests. The host snapshots the source repo before the job
and after it, then grades ``pushed``, ``pr-created`` and the remote part of
the side-effect contract from the two snapshots.

Nothing here runs git on downloaded artifacts. The workspace HEAD is read
as plain files and symlinks are never followed.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from agent_skill_eval.git_state import _run, github_repo_slug, list_review_requests, repo_provider
from agent_skill_eval.models import AssertionResult, GitStateSnapshot

_SHA_RE = re.compile(r"[0-9a-f]{40}")


def _remote_heads(source_repo: str) -> dict[str, str]:
    """Branch name -> SHA on the remote.

    GitHub repos go through ``gh api`` because ``git ls-remote`` runs without
    credentials and sees nothing in a private repo.
    """
    heads: dict[str, str] = {}
    slug = github_repo_slug(source_repo)
    if slug and repo_provider(source_repo) in (None, "github"):
        raw = _run(
            [
                "gh",
                "api",
                "--paginate",
                f"repos/{slug}/git/matching-refs/heads",
                "--jq",
                '.[] | .ref + " " + .object.sha',
            ],
            Path(tempfile.gettempdir()),
        )
        for line in raw.splitlines():
            ref, _, sha = line.partition(" ")
            if ref.startswith("refs/heads/") and sha.strip():
                heads[ref[len("refs/heads/") :]] = sha.strip()
        return heads
    # cwd is an empty temp dir, so no repo-local git config is read.
    with tempfile.TemporaryDirectory(prefix="ase-remote-") as tmp:
        raw = _run(["git", "ls-remote", "--heads", source_repo], Path(tmp))
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].startswith("refs/heads/"):
            heads[parts[1][len("refs/heads/") :]] = parts[0]
    return heads


def snapshot_remote(source_repo: str) -> GitStateSnapshot:
    """Snapshot remote branch heads and PRs/MRs of ``source_repo``."""
    heads = _remote_heads(source_repo)
    return GitStateSnapshot(
        remote_branches=sorted(heads),
        remote_branch_heads=heads,
        open_prs=list_review_requests(source_repo),
    )


def read_workspace_head(workspace: Path) -> Optional[str]:
    """Return the HEAD commit SHA from .git files, without running git."""
    git_dir = workspace / ".git"
    try:
        if git_dir.is_symlink() or not git_dir.is_dir():
            return None
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if _SHA_RE.fullmatch(head):
            return head
        if not head.startswith("ref: refs/heads/"):
            return None
        ref = head[len("ref: ") :]
        if ".." in ref.split("/"):
            return None
        ref_path = git_dir / ref
        if ref_path.is_file() and not ref_path.is_symlink():
            value = ref_path.read_text(encoding="utf-8").strip()
            return value if _SHA_RE.fullmatch(value) else None
        packed = git_dir / "packed-refs"
        if packed.is_file() and not packed.is_symlink():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref and _SHA_RE.fullmatch(parts[0]):
                    return parts[0]
    except (OSError, UnicodeDecodeError):
        return None
    return None


def _open(state: Any) -> bool:
    return str(state or "OPEN").upper() in {"OPEN", "OPENED"}


def attribute_remote_changes(
    pre: GitStateSnapshot,
    post: GitStateSnapshot,
    evidence_text: str,
) -> dict[str, Any]:
    """Attach remote changes to one trial.

    Trials of a job share one remote, so a new branch is attributed to a trial
    only when its name appears in that trial's commands or output. A PR is
    attributed when its head branch is attributed.
    """
    new_branches = sorted(set(post.remote_branches) - set(pre.remote_branches))
    mine = [branch for branch in new_branches if branch and branch in evidence_text]
    pre_numbers = {pr.get("number") for pr in pre.open_prs}
    new_prs = [
        pr
        for pr in post.open_prs
        if pr.get("number") is not None and pr.get("number") not in pre_numbers and pr.get("headRefName") in mine
    ]
    return {
        "branches": mine,
        "heads": {branch: post.remote_branch_heads.get(branch, "") for branch in mine},
        "prs": new_prs,
        "unattributed_branches": [branch for branch in new_branches if branch not in mine],
    }


def check_pushed(assertion: str, changes: dict[str, Any], should_trigger: bool, local_head: Optional[str]):
    branches = changes["branches"]
    if not should_trigger:
        if branches:
            return AssertionResult(
                text=assertion,
                passed=False,
                evidence=f"Skill should not have triggered, but new remote branches appeared: {branches}",
                method="host-remote",
            )
        return AssertionResult(
            text=assertion,
            passed=True,
            evidence="No new remote branches (skill did not trigger)",
            method="host-remote",
        )
    if not branches:
        extra = changes["unattributed_branches"]
        note = f" Unattributed new branches: {extra}" if extra else ""
        return AssertionResult(
            text=assertion,
            passed=False,
            evidence="No new remote branch was pushed by this run." + note,
            method="host-remote",
        )
    heads = changes["heads"]
    if local_head and local_head not in heads.values():
        return AssertionResult(
            text=assertion,
            passed=False,
            evidence=(
                f"Branch(es) {branches} were pushed, but no remote head matches the local HEAD {local_head[:12]}"
            ),
            method="host-remote",
        )
    return AssertionResult(
        text=assertion,
        passed=True,
        evidence=f"Pushed branch(es): {branches}",
        method="host-remote",
    )


def check_pr_created(assertion: str, changes: dict[str, Any], should_trigger: bool) -> AssertionResult:
    prs = [pr for pr in changes["prs"] if _open(pr.get("state"))]
    numbers = [pr.get("number") for pr in changes["prs"]]
    if not should_trigger:
        return AssertionResult(
            text=assertion,
            passed=not changes["prs"],
            evidence=(
                "No new PRs were opened (skill did not trigger)"
                if not changes["prs"]
                else f"Skill should not have triggered, but new PR(s) appeared: {numbers}"
            ),
            method="host-remote",
        )
    if not prs:
        return AssertionResult(
            text=assertion,
            passed=False,
            evidence="No new open PR for a branch pushed by this run",
            method="host-remote",
        )
    return AssertionResult(
        text=assertion,
        passed=True,
        evidence=f"Open PR(s) for pushed branch: {[pr.get('number') for pr in prs]}",
        method="host-remote",
    )


def check_contract(contract: dict[str, Any], changes: dict[str, Any]) -> Optional[AssertionResult]:
    """Grade the remote part of the side-effect contract. None if it has no remote rule."""
    forbid_branches = not contract.get("allow_new_remote_branches", True)
    forbid_reviews = not contract.get("allow_new_review_requests", True)
    if not (forbid_branches or forbid_reviews):
        return None
    problems: list[str] = []
    if forbid_branches and changes["branches"]:
        problems.append("new remote branches: " + ", ".join(changes["branches"]))
    if forbid_reviews and changes["prs"]:
        problems.append("new review requests: " + ", ".join(str(pr.get("number")) for pr in changes["prs"]))
    return AssertionResult(
        text="The run stayed inside its remote side-effect contract",
        passed=not problems,
        evidence="; ".join(problems) or "No remote side-effect violations",
        method="side-effect-contract",
    )


def missing_remote_result(assertion: str) -> AssertionResult:
    return AssertionResult(
        text=assertion,
        passed=False,
        evidence="Remote checks need --source-repo on the Harbor backend; none was given",
        method="host-remote",
    )


__all__ = [
    "attribute_remote_changes",
    "check_contract",
    "check_pr_created",
    "check_pushed",
    "missing_remote_result",
    "read_workspace_head",
    "snapshot_remote",
]
