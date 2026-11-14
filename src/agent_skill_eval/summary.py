"""Iteration-level summary artifacts.

``run`` writes a ``summary.json`` next to ``benchmark.json`` so pass rate,
failures, token totals (cached vs non-cached), cost, budget verdicts, and
cleanup state are available from one file — no shell aggregation over
``timing.json``/``grading.json`` trees. ``agent-skill-eval status`` reads it
back (and can rebuild a reduced version from artifacts for runs that
predate it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

SUMMARY_FILENAME = "summary.json"


def leaf_config_dirs(agent_dir: Path) -> list[Path]:
    """Yield config dirs that directly contain run artifacts.

    Layouts: ``agent/with_skill/outputs`` (single run) or
    ``agent/with_skill/run-N/outputs`` (``--runs N`` > 1).
    """
    leaves: list[Path] = []
    for config_dir in sorted(agent_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        run_dirs = sorted(p for p in config_dir.glob("run-*") if p.is_dir())
        if run_dirs:
            leaves.extend(run_dirs)
        else:
            leaves.append(config_dir)
    return leaves


def _token_totals(timings: list[dict]) -> dict:
    # `or 0` (not a .get default): timing.json fields can be present-but-null.
    input_tokens = sum(t.get("input_tokens") or 0 for t in timings)
    cached = sum(t.get("cached_tokens") or 0 for t in timings)
    output = sum(t.get("output_tokens") or 0 for t in timings)
    total = sum(t.get("total_tokens") or 0 for t in timings)
    # Prefer the per-run value the harness recorded (it knows whether its
    # CLI's input_tokens include cache reads); fall back to the
    # subtraction lower bound only for timing.json files predating it.
    non_cached = 0
    for t in timings:
        nc = t.get("non_cached_input_tokens")
        if nc is None:
            nc = max((t.get("input_tokens") or 0) - (t.get("cached_tokens") or 0), 0)
        non_cached += nc
    # The cached share of everything the model read. non_cached + cached is
    # the only denominator that works for both reporting styles (codex
    # includes cache reads in input_tokens; claude-code/opencode do not).
    all_input = non_cached + cached
    # None when no run reported reasoning telemetry: unknown, not zero.
    reasoning_known = [r for t in timings if (r := t.get("reasoning_output_tokens")) is not None]
    return {
        "input": input_tokens,
        "cached": cached,
        "non_cached_input": non_cached,
        "output": output,
        "reasoning_output": sum(reasoning_known) if reasoning_known else None,
        "total": total,
        "cached_pct": round(cached / all_input, 4) if all_input else 0.0,
    }


def _cost_totals(timings: list[dict]) -> dict:
    costs = [t.get("cost_usd") for t in timings]
    known = [c for c in costs if c is not None]
    return {
        # null means unavailable (no run reported a cost), NOT free.
        "total_usd": round(sum(known), 6) if known else None,
        "runs_with_cost": len(known),
        "runs_total": len(costs),
    }


def _run_full_pass(run: dict) -> bool:
    summary = (run.get("grading") or {}).get("summary") or {}
    return bool(summary.get("total", 0) > 0 and summary.get("failed", 0) == 0 and summary.get("skipped", 0) == 0)


def _stability_summary(completed: list[dict]) -> dict:
    groups: dict[tuple[str, str, bool], list[dict]] = {}
    for run in completed:
        key = (str(run.get("eval_id")), str(run.get("agent")), bool(run.get("with_skill")))
        groups.setdefault(key, []).append(run)

    rows = []
    flaky = []
    for (eval_id, agent, with_skill), runs in sorted(groups.items()):
        full_passes = sum(1 for run in runs if _run_full_pass(run))
        failed = len(runs) - full_passes
        assertion_failures: dict[str, int] = {}
        for run in runs:
            for assertion in (run.get("grading") or {}).get("assertion_results", []):
                if assertion.get("passed") or assertion.get("skipped"):
                    continue
                text = str(assertion.get("text", "assertion"))
                assertion_failures[text] = assertion_failures.get(text, 0) + 1
        row = {
            "eval_id": eval_id,
            "agent": agent,
            "with_skill": with_skill,
            "runs": len(runs),
            "full_passes": full_passes,
            "failures": failed,
            "flaky": full_passes > 0 and failed > 0,
            "failed_assertions": assertion_failures,
        }
        rows.append(row)
        if row["flaky"]:
            flaky.append(row)

    return {
        "groups": rows,
        "flaky_groups": flaky,
        "flaky_count": len(flaky),
    }


def _trigger_rates(completed: list[dict]) -> dict:
    """Skill-trigger rate per config.

    Only runs where the harness could prove invocation or non-invocation
    count in the denominator: unknown telemetry must not count as "did not
    trigger".
    """
    groups: dict[str, dict[str, int]] = {}
    for run in completed:
        config = f"{run.get('agent')}_{'with_skill' if run.get('with_skill') else 'without_skill'}"
        entry = groups.setdefault(config, {"invoked": 0, "known": 0, "unknown": 0, "total": 0})
        entry["total"] += 1
        invoked = run.get("skill_invoked")
        if invoked is None:
            entry["unknown"] += 1
            continue
        entry["known"] += 1
        if invoked:
            entry["invoked"] += 1

    rates = {}
    for config, entry in sorted(groups.items()):
        rates[config] = {
            "rate": round(entry["invoked"] / entry["known"], 4) if entry["known"] else None,
            "invoked": entry["invoked"],
            "known": entry["known"],
            "unknown": entry["unknown"],
            "total": entry["total"],
        }
    return rates


def build_run_summary(
    *,
    skill_name: str,
    iteration: int,
    run_id: str,
    agents: list[str],
    eval_ids: list[str],
    side_effect_levels: list[str],
    runs: int,
    with_baseline: bool,
    duration_seconds: float,
    results: dict[str, dict],
    benchmark: dict,
    cleanup: dict,
    skill_hash: Optional[str] = None,
    budget: Optional[dict] = None,
    hooks: Optional[dict] = None,
    skill_size: Optional[dict] = None,
) -> dict:
    """Build the iteration summary dict written to ``summary.json``."""
    completed = [r for r in results.values() if "timing" in r and "grading" in r]
    errors = [
        {"run": key, "error": r["error"]} for key, r in sorted(results.items()) if "error" in r and not r.get("skipped")
    ]
    skipped = [{"run": key, "reason": r["error"]} for key, r in sorted(results.items()) if r.get("skipped")]

    failed_runs = []
    budget_exceeded = []
    for r in completed:
        grading_summary = r["grading"]["summary"]
        identity = {
            "eval_id": r.get("eval_id"),
            "agent": r.get("agent"),
            "with_skill": r.get("with_skill"),
            "run_index": r.get("run_index", 1),
        }
        if grading_summary.get("failed", 0) > 0:
            failed_runs.append(
                {
                    **identity,
                    "pass_rate": grading_summary.get("pass_rate", 0.0),
                    "side_effect_level": r.get("side_effect_level"),
                    "failed_assertions": [
                        a["text"]
                        for a in r["grading"].get("assertion_results", [])
                        if not a.get("passed") and not a.get("skipped")
                    ],
                }
            )
        if r["timing"].get("budget_exceeded"):
            budget_exceeded.append(
                {
                    **identity,
                    "side_effect_level": r.get("side_effect_level"),
                    "reason": r["timing"].get("budget_reason"),
                }
            )

    timings = [r["timing"] for r in completed]
    pass_rates = {
        config: stats.get("pass_rate", {}).get("mean") for config, stats in benchmark.get("run_summary", {}).items()
    }

    summary = {
        "skill_name": skill_name,
        "skill_hash": skill_hash,
        "iteration": iteration,
        "run_id": run_id,
        "agents": agents,
        "eval_ids": eval_ids,
        "side_effect_levels": side_effect_levels,
        "runs_per_case": runs,
        "with_baseline": with_baseline,
        "duration_seconds": round(duration_seconds, 1),
        "pass_rates": pass_rates,
        "trigger_rate": _trigger_rates(completed),
        "runs_completed": len(completed),
        "runs_failed_grading": len(failed_runs),
        "failed_runs": failed_runs,
        "errors": errors,
        "skipped_runs": skipped,
        "tokens": _token_totals(timings),
        "cost": _cost_totals(timings),
        "stability": _stability_summary(completed),
        "cleanup": {
            "source_repo_provider": cleanup.get("source_repo_provider"),
            "source_repo_project": cleanup.get("source_repo_project") or cleanup.get("source_repo_slug"),
            "remote_branches": cleanup.get("remote_branches", []),
            "pr_numbers": cleanup.get("pr_numbers", []),
            "merge_request_numbers": cleanup.get("merge_request_numbers", []),
            "workspaces": len(cleanup.get("workspaces", [])),
        },
    }
    if skill_size:
        summary["skill_size"] = skill_size
    if budget:
        summary["budget"] = {**budget, "exceeded_runs": budget_exceeded}
    if hooks:
        summary["hooks"] = {
            stage: [
                {"command": rec["command"], "exit_code": rec["exit_code"], "timed_out": rec["timed_out"]}
                for rec in records
            ]
            for stage, records in hooks.items()
            if records
        }
    return summary


def write_run_summary(iteration_dir: Path, summary: dict) -> Path:
    path = iteration_dir / SUMMARY_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def load_summary(iteration_dir: Path) -> Optional[dict]:
    path = iteration_dir / SUMMARY_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def build_summary_from_artifacts(iteration_dir: Path) -> Optional[dict]:
    """Rebuild a reduced summary from saved artifacts (pre-summary runs).

    Walks ``eval-*/<agent>/<config>[/run-N]`` for grading/timing files. Pass
    rates come from ``benchmark.json`` when present. Budget/hook/cleanup
    detail beyond cleanup.json is not reconstructable.
    """
    eval_dirs = sorted(iteration_dir.glob("eval-*"))
    if not eval_dirs:
        return None

    results: dict[str, dict] = {}
    for eval_dir in eval_dirs:
        for agent_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
            for config_dir in leaf_config_dirs(agent_dir):
                grading_path = config_dir / "grading.json"
                timing_path = config_dir / "timing.json"
                if not grading_path.exists() or not timing_path.exists():
                    continue
                try:
                    grading = json.loads(grading_path.read_text())
                    timing = json.loads(timing_path.read_text())
                except (json.JSONDecodeError, OSError):
                    continue

                meta = {}
                meta_path = config_dir / "run_meta.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                    except (json.JSONDecodeError, OSError):
                        meta = {}

                key = f"{eval_dir.name}/{agent_dir.name}/{config_dir.relative_to(agent_dir)}"
                results[key] = {
                    "eval_id": meta.get("eval_id", eval_dir.name[len("eval-") :]),
                    "agent": meta.get("agent", agent_dir.name),
                    "with_skill": meta.get("with_skill", "with_skill" in config_dir.parts),
                    "run_index": meta.get("run_index", 1),
                    "side_effect_level": meta.get("side_effect_level", "local-only"),
                    "skill_hash": meta.get("skill_hash"),
                    "timing": timing,
                    "grading": grading,
                    "skill_invoked": meta.get("skill_invoked"),
                    "skill_invocation_evidence": meta.get("skill_invocation_evidence"),
                }

    if not results:
        return None

    benchmark = {}
    benchmark_path = iteration_dir / "benchmark.json"
    if benchmark_path.exists():
        try:
            benchmark = json.loads(benchmark_path.read_text())
        except (json.JSONDecodeError, OSError):
            benchmark = {}

    cleanup = {}
    cleanup_path = iteration_dir / "cleanup.json"
    if cleanup_path.exists():
        try:
            cleanup = json.loads(cleanup_path.read_text())
        except (json.JSONDecodeError, OSError):
            cleanup = {}

    first = next(iter(results.values()))
    iteration_num = 0
    if iteration_dir.name.startswith("iteration-"):
        try:
            iteration_num = int(iteration_dir.name[len("iteration-") :])
        except ValueError:
            iteration_num = 0

    return build_run_summary(
        skill_name=iteration_dir.parent.name.removesuffix("-workspace"),
        iteration=iteration_num,
        run_id=first.get("run_id", ""),
        agents=sorted({r["agent"] for r in results.values()}),
        eval_ids=sorted({str(r["eval_id"]) for r in results.values()}),
        side_effect_levels=sorted({str(r.get("side_effect_level", "local-only")) for r in results.values()}),
        runs=max(r.get("run_index", 1) for r in results.values()),
        with_baseline=any(not r["with_skill"] for r in results.values()),
        duration_seconds=sum(r["timing"].get("duration_ms", 0) for r in results.values()) / 1000.0,
        results=results,
        benchmark=benchmark,
        cleanup=cleanup,
        skill_hash=first.get("skill_hash"),
    )
