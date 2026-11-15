"""Import Harbor job directories into the native artifact layout."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agent_skill_eval.analysis import safe_eval_dir
from agent_skill_eval.graders import (
    assertion_text,
    classify_assertion,
    grade_assertions,
    summarize_assertion_results,
)
from agent_skill_eval.harbor import harbor_version
from agent_skill_eval.harbor.atif import (
    detect_trigger,
    executed_commands,
    has_trajectory,
    load_trajectory,
    out_of_workspace_access,
    raw_agent_text,
    reasoning_tokens,
    tests_pass_from_trajectory,
)
from agent_skill_eval.harbor.models import HarborExportManifest
from agent_skill_eval.harnesses import _rate
from agent_skill_eval.hooks import apply_post_grade_hooks
from agent_skill_eval.models import AssertionResult, BudgetConfig, GradingResult, RunMeta, TimingData

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


def _command_kind(assertion: Any) -> Optional[str]:
    """Return the structured type for command-evidence assertions, else None."""
    if isinstance(assertion, dict):
        kind = str(assertion.get("type") or "")
    else:
        kind = str(getattr(assertion, "type", "") or "")
    if kind in {"command_ran", "command_not_ran", "tests_pass"}:
        return kind
    if not kind and isinstance(assertion, str) and classify_assertion(assertion) == "command-ran":
        return "command_ran"
    return None


def _apply_pricing(timing: TimingData, model: str, pricing: Optional[dict[str, dict]]) -> None:
    """Fill a missing cost from --pricing-config. Harbor's own cost wins."""
    if timing.cost_usd is not None or not pricing or model not in pricing:
        return
    entry = pricing[model]
    prompt = _rate(entry, "prompt")
    completion = _rate(entry, "completion")
    if prompt <= 0 and completion <= 0:
        return
    non_cached = timing.non_cached_input_tokens
    if non_cached is None:
        non_cached = max(timing.input_tokens - timing.cached_tokens, 0)
    cache_read = _rate(entry, "input_cache_read") or prompt
    timing.cost_usd = non_cached * prompt + timing.cached_tokens * cache_read + timing.output_tokens * completion
    timing.cost_usd_source = "pricing-config"


def _budget_verdict(grading: GradingResult, violations: list[str], budget: Optional[BudgetConfig]) -> GradingResult:
    """Fail the grading when the budget was exceeded (action != warn)."""
    if not violations or budget is None or budget.action == "warn":
        return grading
    results = list(grading.assertion_results) + [
        AssertionResult(
            text="The run stayed within its budget",
            passed=False,
            evidence="budget exceeded: " + "; ".join(violations),
            method="budget",
        )
    ]
    return GradingResult(assertion_results=results, summary=summarize_assertion_results(results))


def _deferred_results(
    case: dict[str, Any],
    *,
    workspace: Path,
    agent_output: str,
    output_dir: Path,
    llm_grader: Any,
    trajectory: Optional[dict[str, Any]] = None,
) -> dict[int, AssertionResult]:
    indices = [int(index) for index in case.get("deferred") or []]
    assertions = case.get("assertions") or []
    results: dict[int, AssertionResult] = {}
    for index in indices:
        if not 0 <= index < len(assertions):
            continue
        kind = _command_kind(assertions[index])
        if kind is not None:
            # Command evidence comes from ATIF only. Without it the result is
            # unknown, never a vacuous pass (command_not_ran) or a guess.
            text = assertion_text(assertions[index])
            if not has_trajectory(trajectory):
                results[index] = AssertionResult(
                    text=text,
                    passed=False,
                    evidence="No ATIF trajectory; command evidence is unavailable",
                    method="deferred",
                    skipped=True,
                )
                continue
            if kind == "tests_pass":
                item = assertions[index]
                command = item.get("command") if isinstance(item, dict) else getattr(item, "command", None)
                passed, evidence = tests_pass_from_trajectory(trajectory, command)
                results[index] = AssertionResult(
                    text=text, passed=passed, evidence=evidence, method="structured:tests_pass"
                )
                continue
        # Grade one assertion per call. grade_assertions reorders its output
        # (deterministic, then LLM, then skipped), so zipping it with the
        # input indices would attach the wrong evidence to the wrong check.
        grading = grade_assertions(
            [assertions[index]],
            agent_output,
            output_dir,
            str(case.get("expected_output") or ""),
            llm_grader,
            workspace=workspace,
            should_trigger=bool(case.get("should_trigger", True)),
        )
        results[index] = (
            grading.assertion_results[0]
            if grading.assertion_results
            else AssertionResult(
                text=str(assertions[index]),
                passed=False,
                evidence="Deferred assertion was not graded",
                method="deferred",
                skipped=True,
            )
        )
    return results


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
    run_id: Optional[str] = None,
    post_grade_commands: Optional[list[str]] = None,
    hook_env: Optional[dict[str, str]] = None,
    budget: Optional[BudgetConfig] = None,
    pricing: Optional[dict[str, dict]] = None,
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
        model = str(agent_config.get("model_name") or (agent_info.get("model_info") or {}).get("name") or "")
        _apply_pricing(timing, model, pricing)
        budget_violations = budget.violations(timing) if budget else []
        if budget_violations and budget:
            timing.budget_exceeded = True
            timing.budget_reason = "; ".join(budget_violations)
        error = _sanity_error(result)
        run_key = f"{case_id}-{agent}-{'with' if with_skill else 'without'}-r{run_index}"
        if error:
            results[run_key] = {"error": error}
            continue

        details = _json(trial_dir / "verifier" / "reward-details.json")
        if not details:
            results[run_key] = {"error": "Verifier result is missing (reward-details.json)"}
            continue
        case = details.get("case") or {}
        if not case:
            fallback = Path(manifest.out_dir) / "tasks" / task_dir_name / "tests" / "case.json"
            case = _json(fallback)
        if not case:
            results[run_key] = {"error": "Verifier did not record the case definition"}
            continue
        if details.get("case_id") and str(details["case_id"]) != str(case.get("case_id")):
            results[run_key] = {"error": "Verifier case_id does not match the exported case"}
            continue
        workdir = str(case.get("workdir") or manifest.workdir)
        workspace = _artifact_workspace(trial_dir, workdir)
        agent_output = raw_agent_text(trial_dir)

        config_name = safe_eval_dir(case_id)
        config_dir = iteration_dir / config_name / agent / ("with_skill" if with_skill else "without_skill")
        if runs > 1:
            config_dir = config_dir / f"run-{run_index}"
        output_dir = config_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Write timing before grading: tests_pass reads timing.json for the
        # exit code. Also save the transcript so `grade` can re-judge without
        # rerunning the agent.
        with open(config_dir / "timing.json", "w", encoding="utf-8") as handle:
            json.dump(timing.model_dump(), handle, indent=2)
        (output_dir / "output.txt").write_text(agent_output, encoding="utf-8")
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
                trajectory=trajectory,
            )
        )

        missing = sorted(set(range(len(case.get("assertions") or []))) - set(by_index))
        if missing:
            results[run_key] = {"error": f"Verifier did not report assertion(s): {missing}"}
            continue

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
        meta = RunMeta(
            eval_id=case.get("case_id", case_id),
            agent=agent,
            with_skill=with_skill,
            iteration=iteration,
            skill_name=manifest.skill_name,
            skill_hash=manifest.skill_hash,
            side_effect_level=str(case.get("side_effect_level") or "local-only"),
            source_repo=None,
            run_id=run_id or job_dir.name,
            run_index=run_index,
            workspace_path=str(workspace) if workspace.exists() else None,
            model=model or None,
            reasoning_effort=(agent_config.get("kwargs") or {}).get("reasoning_effort"),
            base_url=agent_config.get("base_url"),
            harness_version=f"harbor {harbor_version() or manifest.harbor_version or 'unknown'}",
            agent_cli_version=agent_info.get("version"),
            skill_invoked=triggered,
            skill_invocation_evidence=trigger_evidence,
            tags=tags,
        )

        with open(config_dir / "run_meta.json", "w", encoding="utf-8") as handle:
            json.dump(meta.model_dump(), handle, indent=2)
        grading_path = config_dir / "grading.json"
        with open(grading_path, "w", encoding="utf-8") as handle:
            json.dump(grading.model_dump(), handle, indent=2)

        if post_grade_commands:
            env = dict(hook_env or {})
            env.update(
                {
                    "ASE_EVAL_ID": str(case.get("case_id", case_id)),
                    "ASE_AGENT": agent,
                    "ASE_WITH_SKILL": "1" if with_skill else "0",
                    "ASE_RUN_INDEX": str(run_index),
                    "ASE_WORKSPACE_PATH": str(workspace),
                    "ASE_OUTPUT_DIR": str(output_dir),
                    "ASE_TIMING_PATH": str(config_dir / "timing.json"),
                    "ASE_RUN_META_PATH": str(config_dir / "run_meta.json"),
                    "ASE_GRADING_PATH": str(grading_path),
                }
            )
            grading, hook_records = apply_post_grade_hooks(post_grade_commands, env, grading)
            with open(output_dir / "post_grade_hooks.json", "w", encoding="utf-8") as handle:
                json.dump(hook_records, handle, indent=2)

        grading = _budget_verdict(grading, budget_violations, budget)
        with open(grading_path, "w", encoding="utf-8") as handle:
            json.dump(grading.model_dump(), handle, indent=2)
        # Timing is rewritten: pricing and the budget flag were set above.
        with open(config_dir / "timing.json", "w", encoding="utf-8") as handle:
            json.dump(timing.model_dump(), handle, indent=2)
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
            "force_skill_invocation": bool(case.get("force_skill_invocation")),
            "tags": tags,
            "out_of_workspace": out_of_workspace,
        }

    return results
