"""Run a paired skill evaluation as Harbor jobs and import the results."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from agent_skill_eval.harbor import require_harbor
from agent_skill_eval.harbor.export import export_suite
from agent_skill_eval.harbor.importer import import_job
from agent_skill_eval.harbor.models import HarborExportManifest
from agent_skill_eval.models import AgentType
from agent_skill_eval.runner import compute_benchmark
from agent_skill_eval.skills import SkillInstaller
from agent_skill_eval.summary import build_run_summary, write_run_summary

HARBOR_AGENTS = {
    AgentType.OPENCODE: "opencode-v2",
    AgentType.CLAUDE_CODE: "claude-code",
    AgentType.CODEX: "codex",
}
_AGENT_ERROR_OVERRIDES = {
    "NonZeroAgentExitCodeError",
    "AgentTimeoutError",
    "ContextWindowExceededError",
    "OutputTokenExceededError",
    "ContextLengthExceededError",
    "OutputLengthExceededError",
}


@dataclass
class HarborRunOptions:
    evals_path: Path
    skill_path: Path
    workspace: Path
    iteration: int = 1
    agents: list[AgentType] = field(default_factory=lambda: [AgentType.OPENCODE])
    agent_models: dict[AgentType, Optional[str]] = field(default_factory=dict)
    runs: int = 1
    concurrency: int = 1
    with_baseline: bool = True
    grader_model: str = ""
    grader_base_url: Optional[str] = None
    grader_key_file: Optional[Path] = None
    grader_allowed_models: Optional[tuple[str, ...]] = None
    eval_ids: list[str] = field(default_factory=list)
    side_effect_levels: list[str] = field(default_factory=list)
    base_image: str = "python:3.12-slim"
    workdir: str = "/app"
    task_version: str = "1.0"
    harbor_bin: Optional[str] = None
    jobs_dir: Optional[Path] = None
    reasoning_effort: Optional[str] = None
    agent_base_url: Optional[str] = None
    agent_timeout: Optional[int] = None
    agent_retries: Optional[int] = None


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return slug[:80] or "job"


def _agent_name(agent: AgentType) -> str:
    try:
        return HARBOR_AGENTS[agent]
    except KeyError as exc:
        raise ValueError(f"Agent '{agent.value}' has no Harbor adapter. Use opencode, claude-code, or codex.") from exc


def _job_config(
    *,
    job_name: str,
    jobs_dir: Path,
    tasks_dir: Path,
    agent: AgentType,
    model: Optional[str],
    with_skill: bool,
    skill_path: Path,
    runs: int,
    concurrency: int,
    base_url: Optional[str],
    reasoning_effort: Optional[str],
    timeout: Optional[int],
    retries: Optional[int],
) -> dict:
    agent_entry: dict = {"name": _agent_name(agent)}
    if model:
        agent_entry["model_name"] = model
    if base_url:
        agent_entry["base_url"] = base_url
    if timeout:
        agent_entry["override_timeout_sec"] = float(timeout)
    if reasoning_effort:
        agent_entry["kwargs"] = {"reasoning_effort": reasoning_effort}
    if with_skill:
        agent_entry["skills"] = [str(skill_path)]
    return {
        "job_name": job_name,
        "jobs_dir": str(jobs_dir),
        "n_attempts": max(runs, 1),
        "n_concurrent_trials": max(concurrency, 1),
        "agent_setup_timeout_multiplier": 3.0,
        "retry": {"max_retries": max(retries if retries is not None else 1, 0)},
        "override_error_policy": {name: {"has_valid_reward": False} for name in sorted(_AGENT_ERROR_OVERRIDES)},
        "datasets": [
            {
                "path": str(tasks_dir),
                "exclude_task_names": ["*__without"] if with_skill else ["*__with"],
            }
        ],
        "agents": [agent_entry],
        "environment": {"type": "docker"},
    }


def _write_job_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def run_harbor(options: HarborRunOptions) -> Path:
    """Export, run both arms as Harbor jobs, import, and write the summary."""
    harbor_bin = options.harbor_bin or shutil.which("harbor") or "harbor"
    require_harbor()

    missing_models = [agent.value for agent in options.agents if not options.agent_models.get(agent)]
    if missing_models:
        flags = ", ".join(f"--agent-model {name}=<provider/model>" for name in missing_models)
        raise ValueError(f"The Harbor backend needs an explicit model for each agent: {flags}")

    skill_path = options.skill_path.resolve()
    installer = SkillInstaller(skill_path)
    skill_hash = installer.content_hash()
    skill_size = installer.size_metrics()
    skill_name = installer.skill_name
    iteration_dir = options.workspace / f"{skill_name}-workspace" / f"iteration-{options.iteration}"
    harbor_dir = iteration_dir / "harbor"
    run_id = uuid.uuid4().hex[:8]
    # Harbor resumes a job directory whose config matches. A repeat run after
    # a skill or fixture edit must not reuse those trials, so scope the jobs
    # to this run.
    jobs_dir = (options.jobs_dir or harbor_dir / "jobs") / run_id
    export_dir = harbor_dir / "tasks-export"

    manifest: HarborExportManifest = export_suite(
        evals_path=options.evals_path,
        skill_path=skill_path,
        out_dir=export_dir,
        base_image=options.base_image,
        workdir=options.workdir,
        task_version=options.task_version,
        eval_ids=options.eval_ids,
        side_effect_levels=options.side_effect_levels,
        force=True,
    )
    tasks_dir = export_dir / "tasks"
    configs_dir = harbor_dir / "configs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    job_dirs: list[Path] = []

    for agent in options.agents:
        model = options.agent_models.get(agent)
        arms = [True, False] if options.with_baseline else [True]
        for with_skill in arms:
            arm = "with_skill" if with_skill else "without_skill"
            job_name = _slug(f"{skill_name}__{agent.value}__{arm}")
            config = _job_config(
                job_name=job_name,
                jobs_dir=jobs_dir,
                tasks_dir=tasks_dir,
                agent=agent,
                model=model,
                with_skill=with_skill,
                skill_path=skill_path,
                runs=options.runs,
                concurrency=options.concurrency,
                base_url=options.agent_base_url,
                reasoning_effort=options.reasoning_effort,
                timeout=options.agent_timeout,
                retries=options.agent_retries,
            )
            config_path = configs_dir / f"{job_name}.yaml"
            _write_job_config(config_path, config)
            subprocess.run([harbor_bin, "run", "-c", str(config_path), "--yes"], check=True)
            job_dir = jobs_dir / job_name
            if not (job_dir / "result.json").is_file():
                raise RuntimeError(f"Harbor did not write a result for {job_name}: {job_dir}")
            job_dirs.append(job_dir)

    results: dict[str, dict] = {}
    grader = _make_grader(options)
    for job_dir in job_dirs:
        results.update(
            import_job(
                job_dir,
                manifest=manifest,
                iteration_dir=iteration_dir,
                iteration=options.iteration,
                runs=options.runs,
                llm_grader=grader,
            )
        )

    benchmark = compute_benchmark(results, options.agents, options.with_baseline)
    with open(iteration_dir / "benchmark.json", "w", encoding="utf-8") as handle:
        json.dump(benchmark.model_dump(), handle, indent=2)

    cleanup = {
        "source_repo_provider": None,
        "source_repo_project": None,
        "remote_branches": [],
        "pr_numbers": [],
        "merge_request_numbers": [],
        "workspaces": [],
    }
    with open(iteration_dir / "cleanup.json", "w", encoding="utf-8") as handle:
        json.dump(cleanup, handle, indent=2)

    summary = build_run_summary(
        skill_name=skill_name,
        iteration=options.iteration,
        run_id=run_id,
        agents=[agent.value for agent in options.agents],
        eval_ids=[case.case_id for case in manifest.cases],
        side_effect_levels=sorted({case.side_effect_level for case in manifest.cases}),
        runs=options.runs,
        with_baseline=options.with_baseline,
        duration_seconds=time.time() - started,
        results=results,
        benchmark=benchmark.model_dump(),
        cleanup=cleanup,
        skill_hash=skill_hash,
        skill_size=skill_size,
        hooks={},
    )
    write_run_summary(iteration_dir, summary)
    return iteration_dir


def _make_grader(options: HarborRunOptions):
    if not options.grader_model:
        return None
    from agent_skill_eval.graders import LLMGrader

    if not LLMGrader.has_credentials(options.grader_key_file):
        return None
    return LLMGrader(
        model=options.grader_model,
        base_url=options.grader_base_url,
        api_key_file=options.grader_key_file,
        allowed_models=options.grader_allowed_models,
    )
