"""Skill-centric statistics: percentiles, Wilson intervals, and baselines."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Optional

BASELINE_SCHEMA_VERSION = 1


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q between 0 and 1)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * min(max(q, 0.0), 1.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return max(centre - margin, 0.0), min(centre + margin, 1.0)


def _config_name(agent: str, with_skill: bool) -> str:
    return f"{agent}_{'with_skill' if with_skill else 'without_skill'}"


def _case_rates(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Map config -> case id -> full-pass rate from the stability groups."""
    cases: dict[str, dict[str, float]] = {}
    for row in (summary.get("stability") or {}).get("groups") or []:
        config = _config_name(str(row.get("agent")), bool(row.get("with_skill")))
        runs = int(row.get("runs") or 0)
        full_passes = int(row.get("full_passes") or 0)
        rate = full_passes / runs if runs else 0.0
        cases.setdefault(config, {})[str(row.get("eval_id"))] = rate
    return cases


def build_baseline(summary: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for config, stats in (benchmark.get("run_summary") or {}).items():
        cost = stats.get("cost_usd") or {}
        configs[config] = {
            "pass_rate": (stats.get("pass_rate") or {}).get("mean"),
            "pass_hat_k": stats.get("pass_hat_k"),
            "wilson_low": stats.get("pass_hat_k_wilson_low"),
            "wilson_high": stats.get("pass_hat_k_wilson_high"),
            "tokens_median": (stats.get("tokens") or {}).get("median"),
            "tokens_p90": (stats.get("tokens") or {}).get("p90"),
            "time_median": (stats.get("time_seconds") or {}).get("median"),
            "time_p90": (stats.get("time_seconds") or {}).get("p90"),
            "cost_median": cost.get("median"),
            "cost_p90": cost.get("p90"),
            "n": (stats.get("pass_rate") or {}).get("n"),
        }
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "skill_name": summary.get("skill_name"),
        "skill_hash": summary.get("skill_hash"),
        "run_id": summary.get("run_id"),
        "iteration": summary.get("iteration"),
        "agents": summary.get("agents"),
        "configs": configs,
        "cases": _case_rates(summary),
    }


def _bootstrap_delta(
    deltas: list[float],
    *,
    iterations: int = 2000,
    seed: int = 7,
) -> Optional[dict[str, float]]:
    if not deltas:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {
        "mean": sum(deltas) / len(deltas),
        "low": means[int(0.025 * (iterations - 1))],
        "high": means[int(0.975 * (iterations - 1))],
        "n": len(deltas),
    }


def compare_to_baseline(
    summary: dict[str, Any],
    benchmark: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    current_configs = benchmark.get("run_summary") or {}
    base_configs = baseline.get("configs") or {}
    current_cases = _case_rates(summary)
    base_cases = baseline.get("cases") or {}

    configs: dict[str, Any] = {}
    for config, stats in current_configs.items():
        base = base_configs.get(config)
        if not base:
            continue
        current_pass = (stats.get("pass_rate") or {}).get("mean")
        delta_pass = (
            current_pass - base["pass_rate"] if current_pass is not None and base.get("pass_rate") is not None else None
        )
        current_hat = stats.get("pass_hat_k")
        delta_hat = (
            current_hat - base["pass_hat_k"] if current_hat is not None and base.get("pass_hat_k") is not None else None
        )
        configs[config] = {
            "pass_rate": {"current": current_pass, "baseline": base.get("pass_rate"), "delta": delta_pass},
            "pass_hat_k": {"current": current_hat, "baseline": base.get("pass_hat_k"), "delta": delta_hat},
            "tokens_median": (stats.get("tokens") or {}).get("median"),
            "time_median": (stats.get("time_seconds") or {}).get("median"),
            "cost_median": (stats.get("cost_usd") or {}).get("median"),
            "tokens_median_delta": _delta((stats.get("tokens") or {}).get("median"), base.get("tokens_median")),
            "tokens_p90_delta": _delta((stats.get("tokens") or {}).get("p90"), base.get("tokens_p90")),
            "time_median_delta": _delta((stats.get("time_seconds") or {}).get("median"), base.get("time_median")),
            "time_p90_delta": _delta((stats.get("time_seconds") or {}).get("p90"), base.get("time_p90")),
            "cost_median_delta": _delta((stats.get("cost_usd") or {}).get("median"), base.get("cost_median")),
        }

    case_deltas: dict[str, list[float]] = {}
    for config, cases in current_cases.items():
        base = base_cases.get(config) or {}
        shared = sorted(set(cases) & set(base))
        deltas = [cases[case] - base[case] for case in shared]
        if deltas:
            case_deltas[config] = deltas

    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "baseline_skill_hash": baseline.get("skill_hash"),
        "baseline_run_id": baseline.get("run_id"),
        "configs": configs,
        "case_delta": {config: _bootstrap_delta(deltas) for config, deltas in case_deltas.items()},
    }


def _delta(current: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if current is None or baseline is None:
        return None
    return current - baseline


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "## Baseline comparison",
        "",
        f"Baseline skill hash: `{comparison.get('baseline_skill_hash')}` (run `{comparison.get('baseline_run_id')}`)",
        "",
        "| Config | Pass rate | Delta | pass^k | Delta | Tokens median | Delta | "
        "Time median | Delta | Cost median | Delta |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for config, row in (comparison.get("configs") or {}).items():
        pass_row = row["pass_rate"]
        hat_row = row["pass_hat_k"]
        lines.append(
            (
                "| {config} | {pass_cur} | {pass_delta} | {hat_cur} | {hat_delta} | "
                "{tok} | {tok_delta} | {time} | {time_delta} | {cost} | {cost_delta} |"
            ).format(
                config=config,
                pass_cur=_fmt(pass_row["current"]),
                pass_delta=_fmt_delta(pass_row["delta"]),
                hat_cur=_fmt(hat_row["current"]),
                hat_delta=_fmt_delta(hat_row["delta"]),
                tok=_fmt_int(row.get("tokens_median")),
                tok_delta=_fmt_int_delta(row["tokens_median_delta"]),
                time=_fmt_time(row.get("time_median")),
                time_delta=_fmt_time_delta(row["time_median_delta"]),
                cost=_fmt_cost(row.get("cost_median")),
                cost_delta=_fmt_cost_delta(row["cost_median_delta"]),
            )
        )
    lines.extend(["", "Per-case bootstrap deltas:", ""])
    for config, row in (comparison.get("case_delta") or {}).items():
        if row:
            lines.append(
                f"- `{config}`: {row['mean']:+.3f} (95% CI {row['low']:+.3f} to {row['high']:+.3f}, n={row['n']})"
            )
    return "\n".join(lines) + "\n"


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_delta(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _fmt_int(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def _fmt_int_delta(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.0f}"


def _fmt_time(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _fmt_time_delta(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.1f}"


def _fmt_cost(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _fmt_cost_delta(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.4f}"
