"""Import Harbor job directories into the native artifact layout."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agent_skill_eval.graders import grade_assertions, summarize_assertion_results
from agent_skill_eval.harbor import harbor_version
from agent_skill_eval.harbor.atif import (
    detect_trigger,
    executed_commands,
    load_trajectory,
    out_of_workspace_access,
    raw_agent_text,
    reasoning_tokens,
)
from agent_skill_eval.harbor.models import HarborExportManifest
from agent_skill_eval.models import AssertionResult, GradingResult, RunMeta, TimingData

HARBOR_AGENT_MAP = {
    "opencode-v2": "opencode",
    "opencode": "opencode",
    "claude-code": "claude-code",
    "codex": "codex",
}


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _ms(start: Optional[str], finish: Optional[str]) -> int:
    if not start or not finish:
        return 0
    try:
        delta = datetime.fromisoformat(finish.replace("Z", "+00:00")) - datetime.fromisoformat(
            start.replace("Z", "+00:00")
        )
    except ValueError:
        return 0
    return max(int(delta.total_seconds() * 1000), 0)


def _safe_eval_dir(case_id: str) -> str:
    """Keep a hostile eval id from writing outside the iteration directory."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(case_id)).strip("._-") or "case"
    return f"eval-{slug}"


def _artifact_workspace(trial_dir: Path, workdir: str) -> Path:
    relative = workdir.lstrip("/")
    candidates = [
        trial_dir / "artifacts" / relative,
        trial_dir / "artifacts",
        trial_dir / "artifacts" / "app",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return trial_dir / "artifacts"


def _sanity_error(result: dict[str, Any]) -> Optional[str]:
    exception = result.get("exception_info") or {}
    if exception:
        name = str(exception.get("exception_type") or "Exception")
        return f"{name}: {exception.get('exception_message', '')}".strip()
    if not result.get("reward_valid", False):
        return "Harbor marked the reward invalid"
    if result.get("verifier_result") is None:
        return "Harbor did not produce a verifier reward"
    agent_result = result.get("agent_result") or {}
    reported_output = agent_result.get("n_output_tokens")
    if reported_output is not None and int(reported_output) == 0:
        return "Agent produced 0 output tokens"
    if all(agent_result.get(key) is None for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens")):
        return "Harbor reported no token telemetry"
    return None


def _timing(result: dict[str, Any], trajectory: Optional[dict[str, Any]]) -> TimingData:
    agent_result = result.get("agent_result") or {}
    input_tokens = agent_result.get("n_input_tokens")
    cached = agent_result.get("n_cache_tokens")
    output_tokens = agent_result.get("n_output_tokens")
    non_cached = None
    if input_tokens is not None:
        non_cached = max(int(input_tokens) - int(cached or 0), 0)
    exception = result.get("exception_info") or {}
    execution = result.get("agent_execution") or {}
    return TimingData(
        total_tokens=int(input_tokens or 0) + int(output_tokens or 0),
        input_tokens=int(input_tokens or 0),
        output_tokens=int(output_tokens or 0),
        cached_tokens=int(cached or 0),
        non_cached_input_tokens=non_cached,
        reasoning_output_tokens=reasoning_tokens(trajectory),
        cost_usd=agent_result.get("cost_usd"),
        cost_usd_source="harbor" if agent_result.get("cost_usd") is not None else None,
        cost_usd_cli=agent_result.get("cost_usd"),
        duration_ms=_ms(execution.get("started_at"), execution.get("finished_at")),
        exit_code=1 if exception else 0,
        timed_out="Timeout" in str(exception.get("exception_type") or ""),
    )


def _deferred_results(
    case: dict[str, Any],
    *,
    workspace: Path,
    agent_output: str,
    output_dir: Path,
    llm_grader: Any,
) -> dict[int, AssertionResult]:
    indices = [int(index) for index in case.get("deferred") or []]
    assertions = case.get("assertions") or []
    selected = [assertions[index] for index in indices if 0 <= index < len(assertions)]
    if not selected:
        return {}
    grading = grade_assertions(
        selected,
        agent_output,
        output_dir,
        str(case.get("expected_output") or ""),
        llm_grader,
        workspace=workspace,
        should_trigger=bool(case.get("should_trigger", True)),
    )
    return {index: result for index, result in zip(indices, grading.assertion_results)}


def _skill_names(manifest: HarborExportManifest, case: dict[str, Any]) -> list[str]:
    names = {
        manifest.skill_name,
        manifest.skill_frontmatter_name,
        str(case.get("skill_name") or ""),
        str(case.get("skill_frontmatter_name") or ""),
    }
    return sorted(name for name in names if name)


def import_job(
    job_dir: Path,
    *,
    manifest: HarborExportManifest,
    iteration_dir: Path,
    iteration: int,
    runs: int,
    llm_grader: Any = None,
) -> dict[str, dict[str, Any]]:
    """Import one Harbor job and write the native per-run artifacts."""
    results: dict[str, dict[str, Any]] = {}
    counters: dict[tuple[str, str, bool], int] = {}
    trial_dirs = sorted(
        (path for path in job_dir.iterdir() if path.is_dir() and (path / "result.json").is_file()),
        key=lambda path: (str(_json(path / "result.json").get("started_at") or ""), path.name),
    )

    for trial_dir in trial_dirs:
        result = _json(trial_dir / "result.json")
        config = result.get("config") or _json(trial_dir / "config.json")
        agent_config = config.get("agent") or {}
        with_skill = bool(agent_config.get("skills"))
        agent_info = result.get("agent_info") or {}
        agent = HARBOR_AGENT_MAP.get(str(agent_info.get("name") or ""), str(agent_info.get("name") or "unknown"))
        task_name = str(result.get("task_name") or "")
        task_dir_name = Path(str((config.get("task") or {}).get("path") or "")).name
        if not task_dir_name:
            task_dir_name = task_name.split("/")[-1]
        case_id = manifest.tasks.get(task_dir_name, task_dir_name.split("__")[0])
        key = (case_id, agent, with_skill)
        counters[key] = counters.get(key, 0) + 1
        run_index = counters[key]

        trajectory = load_trajectory(trial_dir)
        if trajectory:
            traj_agent = str((trajectory.get("agent") or {}).get("name") or "")
            if traj_agent and agent_info.get("name") and traj_agent != str(agent_info.get("name")):
                trajectory = None
        timing = _timing(result, trajectory)
        error = _sanity_error(result)
        run_key = f"{case_id}-{agent}-{'with' if with_skill else 'without'}-r{run_index}"
        if error:
            results[run_key] = {"error": error}
            continue

        details = _json(trial_dir / "verifier" / "reward-details.json")
        case = details.get("case") or {}
        if not case:
            fallback = Path(manifest.out_dir) / "tasks" / task_dir_name / "tests" / "case.json"
            case = _json(fallback)
        if not case:
            results[run_key] = {"error": "Verifier did not record the case definition"}
            continue
        workdir = str(case.get("workdir") or manifest.workdir)
        workspace = _artifact_workspace(trial_dir, workdir)
        agent_output = raw_agent_text(trial_dir)

        config_name = _safe_eval_dir(str(case_id))
        config_dir = iteration_dir / config_name / agent / ("with_skill" if with_skill else "without_skill")
        if runs > 1:
            config_dir = config_dir / f"run-{run_index}"
        output_dir = config_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "stdout.log", "w", encoding="utf-8") as handle:
            for command in executed_commands(trajectory):
                handle.write(json.dumps({"type": "command_execution", "command": command}) + "\n")

        by_index: dict[int, AssertionResult] = {}
        for entry in details.get("assertions") or []:
            try:
                index = int(entry.get("index"))
            except (TypeError, ValueError):
                continue
            if entry.get("deferred"):
                continue
            by_index[index] = AssertionResult(
                text=str(entry.get("text") or ""),
                passed=bool(entry.get("passed")),
                evidence=str(entry.get("evidence") or ""),
                method=str(entry.get("method") or "deterministic"),
                skipped=bool(entry.get("skipped")),
            )
        by_index.update(
            _deferred_results(
                case,
                workspace=workspace,
                agent_output=agent_output,
                output_dir=output_dir,
                llm_grader=llm_grader,
            )
        )

        ordered = [by_index[index] for index in sorted(by_index)]
        grading = GradingResult(
            assertion_results=ordered,
            summary=summarize_assertion_results(ordered),
        )

        triggered, trigger_evidence = (None, "Skill is not installed in this arm")
        if with_skill:
            triggered, trigger_evidence = detect_trigger(
                trajectory,
                skill_names=_skill_names(manifest, case),
            )

        out_of_workspace = out_of_workspace_access(
            trajectory,
            workdir=workdir,
            extra_allowed_prefixes=("/logs/agent/", "/runtime/skills/", "/root/.harbor/", "/root/.nvm/"),
        )
        tags = list(case.get("tags") or ["correctness"])
        model = str(agent_config.get("model_name") or (agent_info.get("model_info") or {}).get("name") or "")
        meta = RunMeta(
            eval_id=case.get("case_id", case_id),
            agent=agent,
            with_skill=with_skill,
            iteration=iteration,
            skill_name=manifest.skill_name,
            skill_hash=manifest.skill_hash,
            side_effect_level=str(case.get("side_effect_level") or "local-only"),
            source_repo=None,
            run_id=job_dir.name,
            run_index=run_index,
            model=model or None,
            reasoning_effort=(agent_config.get("kwargs") or {}).get("reasoning_effort"),
            base_url=agent_config.get("base_url"),
            harness_version=f"harbor {harbor_version() or manifest.harbor_version or 'unknown'}",
            agent_cli_version=agent_info.get("version"),
            skill_invoked=triggered,
            skill_invocation_evidence=trigger_evidence,
            tags=tags,
        )

        with open(config_dir / "grading.json", "w", encoding="utf-8") as handle:
            json.dump(grading.model_dump(), handle, indent=2)
        with open(config_dir / "timing.json", "w", encoding="utf-8") as handle:
            json.dump(timing.model_dump(), handle, indent=2)
        with open(config_dir / "run_meta.json", "w", encoding="utf-8") as handle:
            json.dump(meta.model_dump(), handle, indent=2)
        with open(output_dir / "harbor_trial.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "job_dir": str(job_dir),
                    "trial_dir": str(trial_dir),
                    "task_dir_name": task_dir_name,
                    "out_of_workspace": out_of_workspace,
                },
                handle,
                indent=2,
            )

        results[run_key] = {
            "eval_id": case.get("case_id", case_id),
            "agent": agent,
            "with_skill": with_skill,
            "run_index": run_index,
            "side_effect_level": str(case.get("side_effect_level") or "local-only"),
            "timing": timing.model_dump(),
            "grading": grading.model_dump(),
            "agent_output": agent_output[:2000],
            "skill_invoked": triggered,
            "skill_invocation_evidence": trigger_evidence,
            "should_trigger": bool(case.get("should_trigger", True)),
            "tags": tags,
            "out_of_workspace": out_of_workspace,
        }

    return results
