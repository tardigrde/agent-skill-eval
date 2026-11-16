from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from agent_skill_eval.models import GitStateSnapshot


def safe_git_env(workspace: Path | None = None) -> dict[str, str]:
    """Environment for running git on untrusted repositories.

    A repository's own config is executable data: ``core.fsmonitor``,
    ``core.hooksPath``, ``diff.external`` and ``core.sshCommand`` can run
    programs. Disable them, ignore system/global config, and trust only the
    workspace path passed in (for git's ownership check).
    """
    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    overrides = {
        "core.fsmonitor": "",
        "core.hooksPath": os.devnull,
        "core.pager": "cat",
        "diff.external": "",
        "core.sshCommand": "false",
        "protocol.ext.allow": "never",
    }
    if workspace is not None:
        overrides["safe.directory"] = str(workspace)
    env["GIT_CONFIG_COUNT"] = str(len(overrides))
    for index, (key, value) in enumerate(overrides.items()):
        env[f"GIT_CONFIG_KEY_{index}"] = key
        env[f"GIT_CONFIG_VALUE_{index}"] = value
    return env


def _run(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=safe_git_env(cwd),
    )
    return result.stdout or ""


def github_repo_slug(source_repo: Optional[str]) -> Optional[str]:
    """Return ``owner/repo`` for GitHub clone URLs or existing slugs."""
    if not source_repo:
        return None

    repo = source_repo.strip().rstrip("/")
    if not repo:
        return None

    if "://" not in repo and ":" in repo:
        host_part, path = repo.split(":", 1)
        if host_part.split("@")[-1].lower() == "github.com":
            return _slug_from_repo_path(path)

    parsed = urlparse(repo)
    if parsed.netloc:
        if (parsed.hostname or "").lower() != "github.com":
            return None
        return _slug_from_repo_path(parsed.path)

    if repo.lower().startswith("github.com/"):
        return _slug_from_repo_path(repo[len("github.com/") :])

    return _slug_from_repo_path(repo)


def repo_provider(source_repo: Optional[str]) -> Optional[str]:
    host = _repo_host(source_repo)
    if host == "github.com":
        return "github"
    if host and "gitlab" in host:
        return "gitlab"
    return None


def repo_project_path(source_repo: Optional[str]) -> Optional[str]:
    if not source_repo:
        return None
    repo = source_repo.strip().rstrip("/")
    if not repo:
        return None
    if "://" not in repo and ":" in repo:
        _host_part, path = repo.split(":", 1)
        return _project_path_from_repo_path(path)

    parsed = urlparse(repo)
    if parsed.netloc:
        return _project_path_from_repo_path(parsed.path)

    if repo.lower().startswith("github.com/"):
        return _project_path_from_repo_path(repo[len("github.com/") :])
    if repo.lower().startswith("gitlab.com/"):
        return _project_path_from_repo_path(repo[len("gitlab.com/") :])

    return _project_path_from_repo_path(repo)


def _repo_host(source_repo: Optional[str]) -> Optional[str]:
    if not source_repo:
        return None
    repo = source_repo.strip().rstrip("/")
    if not repo:
        return None
    if "://" not in repo and ":" in repo:
        host_part, _path = repo.split(":", 1)
        return host_part.split("@")[-1].lower()
    parsed = urlparse(repo)
    if parsed.hostname:
        return parsed.hostname.lower()
    if repo.lower().startswith("github.com/"):
        return "github.com"
    if repo.lower().startswith("gitlab.com/"):
        return "gitlab.com"
    return None


def _slug_from_repo_path(path: str) -> Optional[str]:
    clean = path.split("?", 1)[0].split("#", 1)[0].strip("/").removesuffix(".git")
    parts = [part for part in clean.split("/") if part]
    if len(parts) != 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def _project_path_from_repo_path(path: str) -> Optional[str]:
    clean = path.split("?", 1)[0].split("#", 1)[0].strip("/").removesuffix(".git")
    parts = [part for part in clean.split("/") if part]
    if len(parts) < 2:
        return None
    return "/".join(parts)


def list_review_requests(source_repo: Optional[str]) -> list[dict]:
    """List PRs/MRs (all states) known on the source repo. Empty on any failure."""
    open_prs: list[dict] = []
    if not source_repo:
        return open_prs
    provider = repo_provider(source_repo)
    slug = github_repo_slug(source_repo)
    if slug and provider in (None, "github"):
        prs_raw = _run(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                slug,
                "--state",
                "all",
                "--json",
                "number,headRefName,baseRefName,url,state",
            ],
            Path.cwd(),
        )
        if prs_raw.strip():
            try:
                open_prs = json.loads(prs_raw)
            except json.JSONDecodeError:
                open_prs = []
    elif provider == "gitlab":
        project = repo_project_path(source_repo)
        if project:
            mrs_raw = _run(
                ["glab", "mr", "list", "-R", project, "--all", "--output", "json"],
                Path.cwd(),
            )
            open_prs = _normalize_gitlab_merge_requests(mrs_raw)
    return open_prs


def capture_git_state(workspace: Path, source_repo: Optional[str] = None) -> GitStateSnapshot:
    """Snapshot the git and review-request state of a workspace.

    Captures branches, current HEAD, commit count, recent commits, full SHAs
    for branch heads, remote branch heads, and (if a source repo is
    configured) the list of PRs/MRs currently known on the remote.
    """
    raw_branches = _run(["git", "branch", "-a"], workspace).strip().split("\n")
    local_branches: set[str] = set()
    remote_branches: set[str] = set()
    for line in raw_branches:
        b = line.strip().lstrip("* ")
        if not b or "HEAD" in b:
            continue
        if b.startswith("remotes/"):
            stripped = b[len("remotes/") :]
            if "/" in stripped:
                remote_branches.add(stripped.split("/", 1)[1])
        else:
            local_branches.add(b)

    current = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], workspace).strip()
    head = _run(["git", "rev-parse", "HEAD"], workspace).strip()
    count_raw = _run(["git", "rev-list", "--all", "--count"], workspace).strip()

    sha_log = _run(["git", "log", "--all", "--pretty=format:%H"], workspace).strip()
    commit_shas = [line for line in sha_log.split("\n") if line.strip()]

    short_log = _run(["git", "log", "--all", "--pretty=format:%h %s"], workspace).strip()
    commits = [line for line in short_log.split("\n") if line.strip()]

    branch_heads = dict(
        _parse_for_each_ref(
            _run(
                ["git", "for-each-ref", "refs/heads", "--format=%(refname:short) %(objectname)"],
                workspace,
            )
        )
    )

    remote_for_each = _run(
        ["git", "for-each-ref", "refs/remotes", "--format=%(refname:short) %(objectname)"],
        workspace,
    )
    remote_branch_heads: dict[str, str] = {}
    for refname, sha in _parse_for_each_ref(remote_for_each):
        if "/" in refname:
            short = refname.split("/", 1)[1]
            if short != "HEAD":
                remote_branch_heads[short] = sha

    remotes_raw = _run(["git", "remote", "-v"], workspace).strip()
    remote_names = sorted({line.split()[0] for line in remotes_raw.split("\n") if line.strip()})
    status_lines = [line for line in _run(["git", "status", "--porcelain"], workspace).splitlines() if line.strip()]
    changed_files = sorted({line[3:].strip() for line in status_lines if len(line) >= 4 and line[3:].strip()})

    open_prs = list_review_requests(source_repo)

    return GitStateSnapshot(
        local_branches=sorted(local_branches),
        remote_branches=sorted(remote_branches),
        current_branch=current,
        head_sha=head,
        commit_count=int(count_raw) if count_raw.isdigit() else len(commits),
        commits=commits[:20],
        remote_names=remote_names,
        open_prs=open_prs,
        branch_heads=branch_heads,
        remote_branch_heads=remote_branch_heads,
        commit_shas=commit_shas[:50],
        status_porcelain=status_lines,
        changed_files=changed_files,
    )


def _normalize_gitlab_merge_requests(raw: str) -> list[dict]:
    if not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        values = data.get("merge_requests") or data.get("mergeRequests") or data.get("data") or []
    else:
        values = data
    if not isinstance(values, list):
        return []

    out: list[dict] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        number = _optional_int(item.get("iid") or item.get("number"))
        out.append(
            {
                "number": number,
                "headRefName": item.get("source_branch") or item.get("sourceBranch") or item.get("headRefName"),
                "baseRefName": item.get("target_branch") or item.get("targetBranch") or item.get("baseRefName"),
                "url": item.get("web_url") or item.get("webUrl") or item.get("url"),
                "state": _normalize_gitlab_state(item.get("state")),
            }
        )
    return out


def _normalize_gitlab_state(state: object) -> str:
    raw = str(state or "opened").upper()
    if raw in {"OPEN", "OPENED"}:
        return "OPEN"
    if raw == "MERGED":
        return "MERGED"
    if raw in {"CLOSE", "CLOSED"}:
        return "CLOSED"
    return raw


def _optional_int(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _parse_for_each_ref(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            out.append((parts[0], parts[1]))
    return out


def state_diff(pre: GitStateSnapshot, post: GitStateSnapshot) -> dict:
    """Compute the delta between two state snapshots.

    Returns a dict of the form::

        {
            "new_branches": [...],
            "new_remote_branches": [...],
            "current_branch": str,
            "current_branch_changed": bool,
            "head_advanced": bool,
            "new_commits": [...],
            "new_commit_shas": [...],
            "new_open_prs": [...],
            "eval_branch": str | None,
            "eval_branch_pushed": bool,
            "eval_branch_pushed_matches_head": bool,
        }
    """
    pre_local = set(pre.local_branches)
    pre_remote = set(pre.remote_branches)
    pre_pr_numbers = {p.get("number") for p in pre.open_prs}

    new_branches = sorted(set(post.local_branches) - pre_local)
    new_remote_branches = sorted(set(post.remote_branches) - pre_remote)

    pre_head = pre.head_sha
    post_head = post.head_sha
    head_advanced = bool(pre_head) and bool(post_head) and pre_head != post_head

    pre_sha_set = set(pre.commit_shas)
    new_commit_shas = [s for s in post.commit_shas if s and s not in pre_sha_set]

    new_commits = [c for c in post.commits if c not in pre.commits]

    new_open_prs = [p for p in post.open_prs if p.get("number") not in pre_pr_numbers]

    eval_branch: str | None = None
    if new_branches:
        if post.current_branch and post.current_branch in new_branches:
            eval_branch = post.current_branch
        else:
            eval_branch = new_branches[0]
    elif post.current_branch != pre.current_branch and post.current_branch:
        pre_local_set = set(pre.local_branches)
        pre_remote_set = set(pre.remote_branches)
        if post.current_branch in pre_local_set or post.current_branch in pre_remote_set:
            eval_branch = None
        else:
            eval_branch = post.current_branch

    eval_branch_pushed = False
    eval_branch_pushed_matches_head = False
    if eval_branch and eval_branch in post.remote_branch_heads:
        eval_branch_pushed = True
        pushed_sha = post.remote_branch_heads.get(eval_branch)
        if pushed_sha == post_head:
            eval_branch_pushed_matches_head = True

    return {
        "new_branches": new_branches,
        "new_remote_branches": new_remote_branches,
        "current_branch": post.current_branch,
        "current_branch_changed": pre.current_branch != post.current_branch,
        "head_advanced": head_advanced,
        "new_commits": new_commits,
        "new_commit_shas": new_commit_shas,
        "new_open_prs": new_open_prs,
        "eval_branch": eval_branch,
        "eval_branch_pushed": eval_branch_pushed,
        "eval_branch_pushed_matches_head": eval_branch_pushed_matches_head,
    }
