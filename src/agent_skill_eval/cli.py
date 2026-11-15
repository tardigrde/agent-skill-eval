from __future__ import annotations

import html
import json
import shlex
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from agent_skill_eval import __version__  # noqa: E402
from agent_skill_eval.git_state import repo_provider  # noqa: E402
from agent_skill_eval.graders import (  # noqa: E402
    DEFAULT_GRADER_MODEL,
    OPENWEBUI_GRADER_BASE_URL,
    OPENWEBUI_GRADER_MODEL,
    OPENWEBUI_GRADER_MODELS,
    LLMGrader,
    grade_assertions,
)
from agent_skill_eval.harnesses import HARNESSES, get_cli_version  # noqa: E402
from agent_skill_eval.hooks import HookError  # noqa: E402
from agent_skill_eval.models import SIDE_EFFECT_LEVELS, AgentType, BudgetConfig, CleanupManifest  # noqa: E402
from agent_skill_eval.runner import EvalRunner  # noqa: E402
from agent_skill_eval.skills import SkillInstaller  # noqa: E402
from agent_skill_eval.summary import build_summary_from_artifacts, load_summary  # noqa: E402
from agent_skill_eval.summary import leaf_config_dirs as _leaf_config_dirs  # noqa: E402
from agent_skill_eval.validation import (  # noqa: E402
    EvalSuiteValidationError,
    EvalSuiteValidationReport,
    load_and_validate_suite,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"agent-skill-eval {__version__}")
        raise typer.Exit()


app = typer.Typer(name="agent-skill-eval", help="Evaluate agent skills across OpenCode, Claude Code, Codex, and Fake")
console = Console()
AGENT_CHOICES = ", ".join(t.value for t in AgentType)
DISCOVERY_IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "eval-workspace",
    "node_modules",
    "venv",
}
OPENWEBUI_GRADER_KEY_FILE = Path("/tmp/openwebui.key")


def _suite_has_junit_failure(suite: dict) -> bool:
    summary = suite.get("summary") or {}
    return bool(
        suite.get("status") in {"failed_assertions", "below_threshold"}
        or (summary.get("runs_failed_grading") or 0) > 0
        or summary.get("pass_rates_below_threshold")
        or (summary.get("errors") or 0) > 0
    )


def _run_all_junit_failure_text(suite: dict) -> str:
    summary = suite.get("summary") or {}
    lines = [f"status={suite.get('status', 'unknown')}"]
    if suite.get("error"):
        lines.append(f"error={suite['error']}")
    if suite.get("reason"):
        lines.append(f"reason={suite['reason']}")
    if summary.get("errors"):
        lines.append(f"errors={summary['errors']}")
    below = summary.get("pass_rates_below_threshold") or {}
    for config, rate in sorted(below.items()):
        lines.append(f"below threshold: {config}={_format_pass_rate(rate)}")
    for failed in summary.get("failed_runs") or []:
        config = "with_skill" if failed.get("with_skill") else "without_skill"
        lines.append(
            f"failed run: {failed.get('eval_id')} / {failed.get('agent')} / {config} "
            f"pass_rate={_format_pass_rate(failed.get('pass_rate'))}"
        )
        for assertion in failed.get("failed_assertions") or []:
            lines.append(f"  - {assertion}")
    return "\n".join(lines)


def _write_run_all_junit(manifest: dict, path: Path) -> Path:
    suites = manifest.get("suites") or []
    failures = sum(1 for suite in suites if _suite_has_junit_failure(suite))
    errors = sum(1 for suite in suites if suite.get("status") == "failed")
    skipped = sum(1 for suite in suites if suite.get("status") == "skipped")
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "agent-skill-eval run-all",
            "tests": str(len(suites)),
            "failures": str(failures),
            "errors": str(errors),
            "skipped": str(skipped),
        },
    )
    properties = ET.SubElement(testsuite, "properties")
    ET.SubElement(properties, "property", {"name": "root", "value": str(manifest.get("root", ""))})
    ET.SubElement(properties, "property", {"name": "iteration", "value": str(manifest.get("iteration", ""))})
    ET.SubElement(properties, "property", {"name": "clean", "value": str(manifest.get("clean", ""))})

    for suite in suites:
        testcase = ET.SubElement(
            testsuite,
            "testcase",
            {
                "classname": "agent_skill_eval.run_all",
                "name": suite.get("skill", "unknown"),
                "file": suite.get("evals_path", ""),
            },
        )
        if suite.get("status") == "skipped":
            skipped_el = ET.SubElement(testcase, "skipped", {"message": suite.get("reason", "skipped")})
            skipped_el.text = suite.get("reason", "")
        elif suite.get("status") == "failed":
            error_el = ET.SubElement(testcase, "error", {"message": suite.get("error", "suite failed")})
            error_el.text = _run_all_junit_failure_text(suite)
        elif _suite_has_junit_failure(suite):
            failure_el = ET.SubElement(testcase, "failure", {"message": suite.get("status", "failed")})
            failure_el.text = _run_all_junit_failure_text(suite)

        summary = suite.get("summary") or {}
        output = {
            "status": suite.get("status"),
            "iteration_dir": suite.get("iteration_dir"),
            "pass_rates": summary.get("pass_rates") or {},
        }
        system_out = ET.SubElement(testcase, "system-out")
        system_out.text = json.dumps(output, sort_keys=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(testsuite)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def _changed_paths_since(root: Path, base_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise EvalSuiteValidationError(result.stderr.strip() or f"git diff failed for base ref {base_ref!r}")
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _path_touches_pair(changed_path: str, pair: dict) -> bool:
    skill_dir = str(pair["skill_dir"]).rstrip("/")
    evals_path = str(pair["evals_path"]).rstrip("/") if pair.get("evals_path") else ""
    return (
        changed_path == skill_dir
        or changed_path.startswith(f"{skill_dir}/")
        or (evals_path and (changed_path == evals_path or changed_path.startswith(f"{evals_path}/")))
    )


def _filter_pairs_by_changes(pairs: list[dict], changed_paths: list[str]) -> list[dict]:
    return [pair for pair in pairs if any(_path_touches_pair(path, pair) for path in changed_paths)]


def _build_budget_config(
    *,
    max_input_tokens: Optional[int],
    max_non_cached_input_tokens: Optional[int],
    max_output_tokens: Optional[int],
    max_total_tokens: Optional[int],
    max_reasoning_tokens: Optional[int],
    max_cost_usd: Optional[float],
    max_duration_seconds: Optional[float],
    action: str,
) -> BudgetConfig:
    if action not in ("warn", "fail", "stop-suite"):
        console.print(f"[red]Unknown --budget-action: {action}. Choose warn, fail, or stop-suite.[/red]")
        raise typer.Exit(1)
    return BudgetConfig(
        max_input_tokens=max_input_tokens,
        max_non_cached_input_tokens=max_non_cached_input_tokens,
        max_output_tokens=max_output_tokens,
        max_total_tokens=max_total_tokens,
        max_reasoning_tokens=max_reasoning_tokens,
        max_cost_usd=max_cost_usd,
        max_duration_seconds=max_duration_seconds,
        action=action,
    )


def _agent_model_manifest(agent_model_map: dict[AgentType, str]) -> dict[str, str]:
    return {agent.value: model for agent, model in sorted(agent_model_map.items(), key=lambda item: item[0].value)}


def _grader_settings(
    *,
    grader_model: str,
    grader_base_url: Optional[str],
    grader_key_file: Optional[Path],
    grader_openwebui: bool,
) -> dict:
    model = grader_model if isinstance(grader_model, str) else DEFAULT_GRADER_MODEL
    base_url = grader_base_url if isinstance(grader_base_url, str) else None
    key_file = Path(grader_key_file) if isinstance(grader_key_file, (str, Path)) else None
    openwebui = grader_openwebui is True
    allowed_models: tuple[str, ...] | None = None

    if openwebui and model:
        if model == DEFAULT_GRADER_MODEL:
            model = OPENWEBUI_GRADER_MODEL
        allowed_models = OPENWEBUI_GRADER_MODELS
        if model not in allowed_models:
            console.print(f"[red]--grader-openwebui only allows model {OPENWEBUI_GRADER_MODEL!r}; got {model!r}[/red]")
            raise typer.Exit(1)
        base_url = base_url or OPENWEBUI_GRADER_BASE_URL
        key_file = key_file or OPENWEBUI_GRADER_KEY_FILE

    return {
        "model": model,
        "base_url": base_url,
        "key_file": key_file,
        "allowed_models": allowed_models,
        "openwebui": bool(openwebui and model),
    }


def _missing_grader_credentials_message(key_file: Optional[Path]) -> str:
    if key_file:
        return f"no readable grader API key file at {key_file}"
    return "no OPENROUTER_API_KEY or OPENAI_API_KEY set"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=_version_callback, is_eager=True, help="Show version and exit"
    ),
) -> None:
    """Evaluate agent skills across OpenCode, Claude Code, Codex, and Fake."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command()
def run(
    skill: Path = typer.Option(..., "--skill", "-s", help="Path to skill directory containing SKILL.md"),
    evals: Path = typer.Option(..., "--evals", "-e", help="Path to evals.json file"),
    backend: str = typer.Option(
        "native",
        "--backend",
        help="Execution backend: native (host, not isolated) or harbor (Docker, isolated)",
    ),
    agents: list[str] = typer.Option(
        ["opencode"],
        "--agent",
        "-a",
        help=f"Agent(s) to evaluate: {AGENT_CHOICES}",
    ),
    harbor_base_image: str = typer.Option(
        "python:3.12-slim",
        "--harbor-base-image",
        help="Base image for exported Harbor tasks (must provide git and a POSIX shell)",
    ),
    harbor_workdir: str = typer.Option("/app", "--harbor-workdir", help="Working directory in Harbor tasks"),
    harbor_task_version: str = typer.Option("1.0", "--harbor-task-version", help="Version written to task.toml"),
    harbor_jobs_dir: Optional[Path] = typer.Option(None, "--harbor-jobs-dir", help="Directory for Harbor job results"),
    harbor_bin: Optional[str] = typer.Option(
        None, "--harbor-bin", help="Harbor CLI executable (default: find 'harbor' on PATH)"
    ),
    write_baseline: Optional[Path] = typer.Option(
        None, "--write-baseline", help="Write a baseline JSON for a later --compare"
    ),
    compare_baseline: Optional[Path] = typer.Option(None, "--compare", help="Compare this run to a baseline JSON"),
    workspace: Path = typer.Option(
        Path.cwd() / "eval-workspace",
        "--workspace",
        "-w",
        help="Base directory for eval workspace",
    ),
    iteration: int = typer.Option(1, "--iteration", "-i", help="Iteration number"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of parallel eval runs"),
    baseline: bool = typer.Option(True, "--baseline/--no-baseline", help="Run without-skill baseline"),
    grader_model: str = typer.Option(DEFAULT_GRADER_MODEL, "--grader-model", help="LLM model for rubric grading"),
    grader_base_url: Optional[str] = typer.Option(None, "--grader-base-url", help="Custom API base URL for grader"),
    grader_key_file: Optional[Path] = typer.Option(
        None,
        "--grader-key-file",
        help="Read the LLM grader API key from this file instead of OPENROUTER_API_KEY/OPENAI_API_KEY",
    ),
    grader_openwebui: bool = typer.Option(
        False,
        "--grader-openwebui",
        help="Use the DT OpenWebUI grader endpoint; only allows Gemini 3.1 Flash Lite",
    ),
    source_repo: Optional[str] = typer.Option(
        None, "--source-repo", help="Git repo URL to clone as workspace (instead of fresh git init)"
    ),
    auto_cleanup: bool = typer.Option(
        False,
        "--cleanup",
        help="Auto-cleanup PRs/MRs, branches, and workspaces after run",
    ),
    agent_models: list[str] = typer.Option(
        [],
        "--agent-model",
        "-m",
        help="Model for an agent as 'agent=model' (e.g. 'claude-code=haiku', 'codex=gpt-5-mini'). "
        "A bare 'model' value applies to all selected agents. Repeatable.",
    ),
    harness_base_url: Optional[str] = typer.Option(
        None,
        "--harness-base-url",
        help="API base URL injected into each agent CLI's environment "
        "(ANTHROPIC_BASE_URL for claude-code, OPENAI_BASE_URL for codex/opencode)",
    ),
    runs: int = typer.Option(1, "--runs", "-n", help="Number of runs per (eval, agent, config) for pass@k stats"),
    reasoning_effort: Optional[str] = typer.Option(
        None,
        "--reasoning-effort",
        help="Reasoning effort passed to agents that support it (codex: "
        "model_reasoning_effort, e.g. minimal/low/medium/high/xhigh). Without it, "
        "agents inherit their own local config. Recorded in run_meta.json.",
    ),
    agent_timeout: Optional[int] = typer.Option(
        None, "--timeout", help="Per-run agent timeout in seconds (default 600, env ASE_AGENT_TIMEOUT)"
    ),
    agent_retries: Optional[int] = typer.Option(
        None, "--retries", help="Retries on agent timeout/non-zero exit (default 1, env ASE_AGENT_RETRIES)"
    ),
    eval_ids: list[str] = typer.Option(
        [],
        "--eval-id",
        help="Only run the eval case with this id. Repeatable for multiple cases.",
    ),
    side_effect_levels: list[str] = typer.Option(
        [],
        "--side-effect-level",
        help=f"Only run eval cases with this side-effect level. Repeatable. Choices: {', '.join(SIDE_EFFECT_LEVELS)}",
    ),
    max_input_tokens_per_case: Optional[int] = typer.Option(
        None, "--max-input-tokens-per-case", help="Budget: max input tokens per run (checked after the run)"
    ),
    max_non_cached_input_tokens_per_case: Optional[int] = typer.Option(
        None,
        "--max-non-cached-input-tokens-per-case",
        help="Budget: max non-cached (billable-rate) input tokens per run (checked after the run)",
    ),
    max_output_tokens_per_case: Optional[int] = typer.Option(
        None, "--max-output-tokens-per-case", help="Budget: max output tokens per run (checked after the run)"
    ),
    max_reasoning_tokens_per_case: Optional[int] = typer.Option(
        None,
        "--max-reasoning-tokens-per-case",
        help="Budget: max reasoning output tokens per run; only enforced for "
        "agents that report reasoning telemetry (checked after the run)",
    ),
    max_total_tokens_per_case: Optional[int] = typer.Option(
        None, "--max-total-tokens-per-case", help="Budget: max total tokens per run (checked after the run)"
    ),
    max_cost_per_case: Optional[float] = typer.Option(
        None, "--max-cost-per-case", help="Budget: max USD cost per run (needs a cost source; checked after the run)"
    ),
    max_duration_per_case: Optional[float] = typer.Option(
        None,
        "--max-duration-per-case",
        help="Budget: max seconds per run including retries (post-run check; --timeout still kills mid-run)",
    ),
    budget_action: str = typer.Option(
        "fail",
        "--budget-action",
        help="What a budget violation does: warn (print only), fail (failed 'budget' "
        "assertion in grading), stop-suite (fail + skip runs not yet started)",
    ),
    pre_run_commands: list[str] = typer.Option(
        [],
        "--pre-run-command",
        help="Shell command run once before any agent case (setup/preflight). "
        "Non-zero exit aborts the suite before any model call. Repeatable. "
        "Receives ASE_* run metadata env vars.",
    ),
    post_grade_commands: list[str] = typer.Option(
        [],
        "--post-grade-command",
        help="Shell command run per case after grading (external side-effect grader). "
        "stdout JSON [{text, passed, evidence}] is merged into grading.json; "
        "non-zero exit adds a failed check. Repeatable. Receives ASE_* env vars.",
    ),
    post_run_commands: list[str] = typer.Option(
        [],
        "--post-run-command",
        help="Shell command run once after the suite (teardown). Failures are "
        "recorded in summary.json but don't fail the run. Repeatable.",
    ),
    pricing_config: Optional[Path] = typer.Option(
        None,
        "--pricing-config",
        help="JSON file mapping model id -> {prompt, completion, input_cache_read, "
        "input_cache_write} USD-per-token rates, used to compute cost_usd for "
        "agents whose CLI reports no cost (e.g. codex)",
    ),
):
    """Run skill evaluations."""
    if not skill.exists():
        console.print(f"[red]Skill path not found: {skill}[/red]")
        raise typer.Exit(1)
    if not evals.exists():
        console.print(f"[red]Evals file not found: {evals}[/red]")
        raise typer.Exit(1)

    agent_types = _parse_agent_type_options(agents)

    agent_model_map = _parse_agent_models(agent_models, agent_types)

    from agent_skill_eval.skills import SkillInstaller

    try:
        installer = SkillInstaller(skill)
        for problem in installer.frontmatter_problems():
            console.print(f"[yellow]Warning: {problem}[/yellow]")
        for warning in installer.frontmatter_warnings():
            console.print(f"[yellow]Warning: {warning}[/yellow]")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    grader = _grader_settings(
        grader_model=grader_model,
        grader_base_url=grader_base_url,
        grader_key_file=grader_key_file,
        grader_openwebui=grader_openwebui,
    )

    if grader["model"] and not LLMGrader.has_credentials(grader["key_file"]):
        missing = _missing_grader_credentials_message(grader["key_file"])
        console.print(
            f"[yellow]Warning: {missing}. Assertions that need LLM rubric grading "
            "will be SKIPPED (not failed).[/yellow]"
        )

    budget = _build_budget_config(
        max_input_tokens=max_input_tokens_per_case,
        max_non_cached_input_tokens=max_non_cached_input_tokens_per_case,
        max_output_tokens=max_output_tokens_per_case,
        max_total_tokens=max_total_tokens_per_case,
        max_reasoning_tokens=max_reasoning_tokens_per_case,
        max_cost_usd=max_cost_per_case,
        max_duration_seconds=max_duration_per_case,
        action=budget_action,
    )

    pricing = _load_pricing_config(pricing_config)

    if backend not in {"native", "harbor"}:
        console.print(f"[red]Unknown backend: {backend}. Use native or harbor.[/red]")
        raise typer.Exit(2)

    if backend == "harbor":
        from agent_skill_eval.harbor.run import HarborRunOptions, run_harbor

        if AgentType.FAKE in agent_types:
            console.print("[red]The fake harness has no Harbor adapter. Choose opencode, claude-code, or codex.[/red]")
            raise typer.Exit(1)
        if source_repo:
            console.print("[red]--source-repo is not supported with --backend harbor.[/red]")
            raise typer.Exit(1)
        if pre_run_commands or post_run_commands or post_grade_commands:
            console.print("[yellow]Warning: run hooks are not applied with --backend harbor.[/yellow]")
        if pricing_config:
            console.print("[yellow]Warning: --pricing-config is not applied with --backend harbor.[/yellow]")
        budget_requested = any(
            value is not None
            for value in (
                max_input_tokens_per_case,
                max_non_cached_input_tokens_per_case,
                max_output_tokens_per_case,
                max_reasoning_tokens_per_case,
                max_total_tokens_per_case,
                max_cost_per_case,
                max_duration_per_case,
            )
        )
        if budget_requested:
            console.print("[yellow]Warning: budget checks are not applied with --backend harbor.[/yellow]")

        try:
            result_dir = run_harbor(
                HarborRunOptions(
                    evals_path=evals,
                    skill_path=skill,
                    workspace=workspace,
                    iteration=iteration,
                    agents=agent_types,
                    agent_models=agent_model_map,
                    runs=runs,
                    concurrency=concurrency,
                    with_baseline=baseline,
                    grader_model=grader["model"],
                    grader_base_url=grader["base_url"],
                    grader_key_file=grader["key_file"],
                    grader_allowed_models=grader["allowed_models"],
                    eval_ids=eval_ids,
                    side_effect_levels=side_effect_levels,
                    base_image=harbor_base_image,
                    workdir=harbor_workdir,
                    task_version=harbor_task_version,
                    harbor_bin=harbor_bin,
                    jobs_dir=harbor_jobs_dir,
                    reasoning_effort=reasoning_effort,
                    agent_base_url=harness_base_url,
                    agent_timeout=agent_timeout,
                    agent_retries=agent_retries,
                )
            )
        except (RuntimeError, ValueError, subprocess.CalledProcessError) as e:
            console.print(f"[red]Harbor run failed: {e}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold green]Done! Results in: {result_dir}[/bold green]")
        _apply_analysis_options(result_dir, write_baseline, compare_baseline)
        if auto_cleanup:
            console.print()
            console.print("[yellow]Running cleanup...[/yellow]")
            _cleanup_iteration(result_dir, workspace)
            console.print("[green]Cleanup complete![/green]")
        return

    try:
        runner = EvalRunner(
            skill_path=skill,
            evals_path=evals,
            workspace_base=workspace,
            agents=agent_types,
            concurrency=concurrency,
            with_baseline=baseline,
            grader_model=grader["model"],
            grader_base_url=grader["base_url"],
            grader_key_file=grader["key_file"],
            grader_allowed_models=grader["allowed_models"],
            source_repo=source_repo,
            agent_models=agent_model_map,
            harness_base_url=harness_base_url,
            agent_timeout=agent_timeout,
            agent_max_retries=agent_retries,
            runs=runs,
            eval_ids=eval_ids,
            side_effect_levels=side_effect_levels,
            budget=budget,
            pricing=pricing,
            pre_run_commands=pre_run_commands,
            post_grade_commands=post_grade_commands,
            post_run_commands=post_run_commands,
            reasoning_effort=reasoning_effort,
        )
    except (ValueError, EvalSuiteValidationError) as e:
        # Unknown --eval-id, invalid evals.json, etc.: fail before any model call.
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    _print_validation_report(runner.validation_report)

    try:
        result_dir = runner.run(iteration)
    except HookError as e:
        console.print(f"[red]{e}[/red]")
        console.print("[red]Suite aborted by pre-run hook; no agent was run.[/red]")
        raise typer.Exit(1)
    console.print(f"\n[bold green]Done! Results in: {result_dir}[/bold green]")
    _apply_analysis_options(result_dir, write_baseline, compare_baseline)

    if auto_cleanup:
        console.print()
        console.print("[yellow]Running cleanup...[/yellow]")
        _cleanup_iteration(result_dir, workspace)
        console.print("[green]Cleanup complete![/green]")


def _apply_analysis_options(
    result_dir: Path,
    write_baseline: Optional[Path],
    compare_baseline: Optional[Path],
) -> None:
    """Write and compare baselines after a run, for either backend."""
    if not write_baseline and not compare_baseline:
        return

    from agent_skill_eval.analysis import (
        build_baseline,
        compare_to_baseline,
        read_json,
        render_comparison_markdown,
        write_json,
    )

    summary_path = result_dir / "summary.json"
    benchmark_path = result_dir / "benchmark.json"
    if not summary_path.is_file() or not benchmark_path.is_file():
        console.print(f"[yellow]No summary/benchmark in {result_dir}; analysis skipped.[/yellow]")
        return
    summary = read_json(summary_path)
    benchmark = read_json(benchmark_path)

    if write_baseline:
        write_json(write_baseline, build_baseline(summary, benchmark))
        console.print(f"[green]Baseline written to {write_baseline}[/green]")

    if compare_baseline:
        if not compare_baseline.is_file():
            console.print(f"[red]Baseline not found: {compare_baseline}[/red]")
            return
        comparison = compare_to_baseline(summary, benchmark, read_json(compare_baseline))
        write_json(result_dir / "comparison.json", comparison)
        markdown = render_comparison_markdown(comparison)
        (result_dir / "comparison.md").write_text(markdown, encoding="utf-8")
        console.print(f"\n[bold]Baseline comparison[/bold] (written to {result_dir / 'comparison.md'})")
        console.print(markdown)


@app.command(name="export-harbor")
def export_harbor(
    evals: Path = typer.Option(..., "--evals", "-e", help="Path to evals.json file"),
    skill: Path = typer.Option(..., "--skill", "-s", help="Path to skill directory containing SKILL.md"),
    out: Path = typer.Option(..., "--out", "-o", help="Directory for the exported Harbor tasks"),
    base_image: str = typer.Option(
        "python:3.12-slim",
        "--base-image",
        help="Base image for the agent environment (must provide a POSIX shell and git)",
    ),
    workdir: str = typer.Option("/app", "--workdir", help="Working directory inside the task container"),
    task_version: str = typer.Option("1.0", "--task-version", help="Semantic version written to each task.toml"),
    eval_ids: list[str] = typer.Option([], "--eval-id", help="Only export this eval id. Repeatable."),
    side_effect_levels: list[str] = typer.Option(
        [],
        "--side-effect-level",
        help="Only export eval cases with this side-effect level. Repeatable. "
        f"Choices: {', '.join(SIDE_EFFECT_LEVELS)}",
    ),
    force: bool = typer.Option(False, "--force", help="Replace a non-empty output directory"),
) -> None:
    """Export an eval suite as a directory of Harbor tasks."""
    from agent_skill_eval.harbor.export import export_suite

    try:
        manifest = export_suite(
            evals_path=evals,
            skill_path=skill,
            out_dir=out,
            base_image=base_image,
            workdir=workdir,
            task_version=task_version,
            eval_ids=eval_ids,
            side_effect_levels=side_effect_levels,
            force=force,
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    task_count = len(manifest.tasks)
    console.print(f"[green]Exported {len(manifest.cases)} case(s) as {task_count} Harbor task(s) to {out}[/green]")
    console.print(f"Skill: [bold]{manifest.skill_name}[/bold] (hash {manifest.skill_hash[:12]})")
    console.print(
        f"Next: run [bold]harbor lint[/bold] on each task under {out / 'tasks'}, then start the two arm jobs."
    )


@app.command(name="run-all")
def run_all(
    root: Path = typer.Option(Path.cwd(), "--root", "-r", help="Directory to recursively search for paired suites"),
    agents: list[str] = typer.Option(
        ["opencode"],
        "--agent",
        "-a",
        help=f"Agent(s) to evaluate: {AGENT_CHOICES}",
    ),
    workspace: Path = typer.Option(
        Path.cwd() / "eval-workspace",
        "--workspace",
        "-w",
        help="Base directory for eval workspaces and aggregate manifest",
    ),
    iteration: int = typer.Option(1, "--iteration", "-i", help="Iteration number"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of parallel eval runs within each suite"),
    suite_concurrency: int = typer.Option(
        1,
        "--suite-concurrency",
        help="Number of discovered skill suites to run concurrently",
    ),
    baseline: bool = typer.Option(True, "--baseline/--no-baseline", help="Run without-skill baseline"),
    grader_model: str = typer.Option(DEFAULT_GRADER_MODEL, "--grader-model", help="LLM model for rubric grading"),
    grader_base_url: Optional[str] = typer.Option(None, "--grader-base-url", help="Custom API base URL for grader"),
    grader_key_file: Optional[Path] = typer.Option(
        None,
        "--grader-key-file",
        help="Read the LLM grader API key from this file instead of OPENROUTER_API_KEY/OPENAI_API_KEY",
    ),
    grader_openwebui: bool = typer.Option(
        False,
        "--grader-openwebui",
        help="Use the DT OpenWebUI grader endpoint; only allows Gemini 3.1 Flash Lite",
    ),
    source_repo: Optional[str] = typer.Option(
        None, "--source-repo", help="Git repo URL to clone as workspace for each suite"
    ),
    agent_models: list[str] = typer.Option(
        [],
        "--agent-model",
        "-m",
        help="Model for an agent as 'agent=model'. A bare value applies to all selected agents. Repeatable.",
    ),
    harness_base_url: Optional[str] = typer.Option(None, "--harness-base-url", help="Agent CLI API base URL"),
    runs: int = typer.Option(1, "--runs", "-n", help="Number of runs per (eval, agent, config)"),
    side_effect_levels: list[str] = typer.Option(
        [],
        "--side-effect-level",
        help=f"Only run eval cases with this side-effect level. Repeatable. Choices: {', '.join(SIDE_EFFECT_LEVELS)}",
    ),
    agent_timeout: Optional[int] = typer.Option(None, "--timeout", help="Per-run agent timeout in seconds"),
    agent_retries: Optional[int] = typer.Option(None, "--retries", help="Retries on agent timeout/non-zero exit"),
    reasoning_effort: Optional[str] = typer.Option(None, "--reasoning-effort", help="Reasoning effort for Codex"),
    max_input_tokens_per_case: Optional[int] = typer.Option(
        None, "--max-input-tokens-per-case", help="Budget: max input tokens per run (checked after the run)"
    ),
    max_non_cached_input_tokens_per_case: Optional[int] = typer.Option(
        None,
        "--max-non-cached-input-tokens-per-case",
        help="Budget: max non-cached input tokens per run (checked after the run)",
    ),
    max_output_tokens_per_case: Optional[int] = typer.Option(
        None, "--max-output-tokens-per-case", help="Budget: max output tokens per run (checked after the run)"
    ),
    max_reasoning_tokens_per_case: Optional[int] = typer.Option(
        None,
        "--max-reasoning-tokens-per-case",
        help="Budget: max reasoning output tokens per run when telemetry is available",
    ),
    max_total_tokens_per_case: Optional[int] = typer.Option(
        None, "--max-total-tokens-per-case", help="Budget: max total tokens per run (checked after the run)"
    ),
    max_cost_per_case: Optional[float] = typer.Option(
        None, "--max-cost-per-case", help="Budget: max USD cost per run (needs a cost source)"
    ),
    max_duration_per_case: Optional[float] = typer.Option(
        None, "--max-duration-per-case", help="Budget: max seconds per run including retries"
    ),
    budget_action: str = typer.Option(
        "fail",
        "--budget-action",
        help="What a budget violation does: warn, fail, or stop-suite",
    ),
    pre_run_commands: list[str] = typer.Option(
        [],
        "--pre-run-command",
        help="Shell command run once before each discovered suite. Repeatable.",
    ),
    post_grade_commands: list[str] = typer.Option(
        [],
        "--post-grade-command",
        help="Shell command run per case after grading for each discovered suite. Repeatable.",
    ),
    post_run_commands: list[str] = typer.Option(
        [],
        "--post-run-command",
        help="Shell command run once after each discovered suite. Repeatable.",
    ),
    pricing_config: Optional[Path] = typer.Option(
        None,
        "--pricing-config",
        help="JSON file mapping model id to USD-per-token rates for cost budgets",
    ),
    fail_on_assertions: bool = typer.Option(
        False,
        "--fail-on-assertions/--no-fail-on-assertions",
        help="Exit non-zero when any completed run has failed assertions",
    ),
    min_pass_rate: Optional[float] = typer.Option(
        None,
        "--min-pass-rate",
        help="Exit non-zero when any suite config pass rate is below this threshold (0.0 to 1.0)",
    ),
    junit_xml: Optional[Path] = typer.Option(
        None,
        "--junit-xml",
        help="Write a JUnit XML report for the aggregate run-all result",
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only",
        help="Validate and write the run-all manifest without running any agent suites",
    ),
    changed_only: bool = typer.Option(
        False,
        "--changed-only/--all-suites",
        help="Only run paired suites whose skill or eval files changed since --base-ref",
    ),
    base_ref: str = typer.Option("origin/main", "--base-ref", help="Git ref to diff against for --changed-only"),
):
    """Run every paired skill/eval suite discovered under a repo."""
    started = time.time()
    if min_pass_rate is not None and not 0 <= min_pass_rate <= 1:
        console.print("[red]--min-pass-rate must be between 0.0 and 1.0[/red]")
        raise typer.Exit(1)
    if suite_concurrency < 1:
        console.print("[red]--suite-concurrency must be at least 1[/red]")
        raise typer.Exit(1)
    agent_types = _parse_agent_type_options(agents)
    agent_model_map = _parse_agent_models(agent_models, agent_types)
    grader = _grader_settings(
        grader_model=grader_model,
        grader_base_url=grader_base_url,
        grader_key_file=grader_key_file,
        grader_openwebui=grader_openwebui,
    )
    budget = _build_budget_config(
        max_input_tokens=max_input_tokens_per_case,
        max_non_cached_input_tokens=max_non_cached_input_tokens_per_case,
        max_output_tokens=max_output_tokens_per_case,
        max_total_tokens=max_total_tokens_per_case,
        max_reasoning_tokens=max_reasoning_tokens_per_case,
        max_cost_usd=max_cost_per_case,
        max_duration_seconds=max_duration_per_case,
        action=budget_action,
    )
    pricing = _load_pricing_config(pricing_config)
    if grader["model"] and not LLMGrader.has_credentials(grader["key_file"]):
        missing = _missing_grader_credentials_message(grader["key_file"])
        console.print(
            f"[yellow]Warning: {missing}. Assertions that need LLM rubric grading "
            "will be SKIPPED (not failed).[/yellow]"
        )

    suites = _discover_eval_suites(root)
    skills = _discover_skills(root)
    pairs = [
        pair for pair in _skill_eval_pairs(root, suites, skills) if pair["status"] == "paired" and pair["evals_path"]
    ]
    changed_paths: list[str] = []
    if changed_only:
        try:
            changed_paths = _changed_paths_since(root, base_ref)
        except EvalSuiteValidationError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
        pairs = _filter_pairs_by_changes(pairs, changed_paths)

    if not pairs:
        if changed_only:
            workspace.mkdir(parents=True, exist_ok=True)
            aggregate = {
                "root": str(root),
                "iteration": iteration,
                "side_effect_levels": side_effect_levels,
                "suite_concurrency": suite_concurrency,
                "agents": [agent.value for agent in agent_types],
                "agent_models": _agent_model_manifest(agent_model_map),
                "harness_base_url": harness_base_url,
                "reasoning_effort": reasoning_effort,
                "timeout": agent_timeout,
                "retries": agent_retries,
                "runs": runs,
                "baseline": baseline,
                "grader_model": grader["model"],
                "grader_base_url": grader["base_url"],
                "grader_key_file": str(grader["key_file"]) if grader["key_file"] else None,
                "grader_openwebui": grader["openwebui"],
                "plan_only": plan_only,
                "changed_only": True,
                "base_ref": base_ref,
                "changed_paths": changed_paths,
                "source_repo": source_repo,
                "budget": budget.model_dump() if budget.enabled else None,
                "pricing_config": str(pricing_config) if pricing_config else None,
                "hook_counts": {
                    "pre_run": len(pre_run_commands),
                    "post_grade": len(post_grade_commands),
                    "post_run": len(post_run_commands),
                },
                "suites": [],
                "failures": 0,
                "skipped": 0,
                "runs_failed_grading": 0,
                "suites_failed_grading": 0,
                "suites_below_min_pass_rate": 0,
                "min_pass_rate": min_pass_rate,
                "duration_seconds": round(time.time() - started, 1),
                "clean": True,
            }
            manifest_path = workspace / f"run-all-iteration-{iteration}.json"
            manifest_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
            console.print(f"[yellow]No paired skill/eval suites changed since {base_ref}[/yellow]")
            console.print(f"[green]Wrote aggregate manifest: {manifest_path}[/green]")
            if junit_xml:
                junit_path = _write_run_all_junit(aggregate, junit_xml)
                console.print(f"[green]Wrote JUnit XML report: {junit_path}[/green]")
            return
        console.print(f"[red]No paired skill/eval suites found under {root}[/red]")
        raise typer.Exit(1)

    workspace.mkdir(parents=True, exist_ok=True)
    aggregate: dict = {
        "root": str(root),
        "iteration": iteration,
        "side_effect_levels": side_effect_levels,
        "suite_concurrency": suite_concurrency,
        "agents": [agent.value for agent in agent_types],
        "agent_models": _agent_model_manifest(agent_model_map),
        "harness_base_url": harness_base_url,
        "reasoning_effort": reasoning_effort,
        "timeout": agent_timeout,
        "retries": agent_retries,
        "runs": runs,
        "baseline": baseline,
        "grader_model": grader["model"],
        "grader_base_url": grader["base_url"],
        "grader_key_file": str(grader["key_file"]) if grader["key_file"] else None,
        "grader_openwebui": grader["openwebui"],
        "plan_only": plan_only,
        "changed_only": changed_only,
        "base_ref": base_ref if changed_only else None,
        "changed_paths": changed_paths,
        "source_repo": source_repo,
        "budget": budget.model_dump() if budget.enabled else None,
        "pricing_config": str(pricing_config) if pricing_config else None,
        "hook_counts": {
            "pre_run": len(pre_run_commands),
            "post_grade": len(post_grade_commands),
            "post_run": len(post_run_commands),
        },
        "suites": [],
        "failures": 0,
        "skipped": 0,
        "runs_failed_grading": 0,
        "suites_failed_grading": 0,
        "suites_below_min_pass_rate": 0,
        "min_pass_rate": min_pass_rate,
    }

    if plan_only:
        aggregate["planned_suites"] = 0
        aggregate["planned_cases"] = 0
        for pair in pairs:
            try:
                suite, _ = load_and_validate_suite(root / pair["evals_path"], side_effect_levels=side_effect_levels)
            except EvalSuiteValidationError as e:
                message = str(e)
                if message.startswith("No evals matched side-effect level filter"):
                    aggregate["skipped"] += 1
                    aggregate["suites"].append(
                        {
                            "status": "skipped",
                            "skill": pair["skill"],
                            "skill_dir": pair["skill_dir"],
                            "evals_path": pair["evals_path"],
                            "reason": message,
                        }
                    )
                    continue
                aggregate["failures"] += 1
                aggregate["suites"].append(
                    {
                        "status": "failed",
                        "skill": pair["skill"],
                        "skill_dir": pair["skill_dir"],
                        "evals_path": pair["evals_path"],
                        "error": message,
                    }
                )
                continue

            aggregate["planned_suites"] += 1
            aggregate["planned_cases"] += len(suite.evals)
            aggregate["suites"].append(
                {
                    "status": "planned",
                    "skill": pair["skill"],
                    "skill_dir": pair["skill_dir"],
                    "evals_path": pair["evals_path"],
                    "planned_cases": len(suite.evals),
                }
            )

        aggregate["duration_seconds"] = round(time.time() - started, 1)
        aggregate["clean"] = aggregate["failures"] == 0
        manifest_path = workspace / f"run-all-iteration-{iteration}.json"
        manifest_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote run-all plan manifest: {manifest_path}[/green]")
        if junit_xml:
            junit_path = _write_run_all_junit(aggregate, junit_xml)
            console.print(f"[green]Wrote JUnit XML report: {junit_path}[/green]")
        if aggregate["failures"]:
            console.print(f"[red]{aggregate['failures']} suite(s) failed validation[/red]")
            raise typer.Exit(1)
        return

    def _suite_result(index: int, pair: dict) -> dict:
        base = {
            "index": index,
            "failures": 0,
            "skipped": 0,
            "runs_failed_grading": 0,
            "suites_failed_grading": 0,
            "suites_below_min_pass_rate": 0,
        }
        skill_path = root / pair["skill_dir"]
        evals_path = root / pair["evals_path"]
        try:
            load_and_validate_suite(evals_path, side_effect_levels=side_effect_levels)
        except EvalSuiteValidationError as e:
            message = str(e)
            if message.startswith("No evals matched side-effect level filter"):
                entry = {
                    "status": "skipped",
                    "skill": pair["skill"],
                    "skill_dir": pair["skill_dir"],
                    "evals_path": pair["evals_path"],
                    "reason": message,
                }
                console.print(f"[yellow]Skipped {pair['skill']}: {message}[/yellow]")
                return {**base, "skipped": 1, "entry": entry}
            entry = {
                "status": "failed",
                "skill": pair["skill"],
                "skill_dir": pair["skill_dir"],
                "evals_path": pair["evals_path"],
                "error": message,
            }
            console.print(f"[red]Validation failed for {pair['skill']}: {message}[/red]")
            return {**base, "failures": 1, "entry": entry}

        console.print(f"[cyan]Running {pair['skill']} ({pair['evals_path']})[/cyan]")
        try:
            runner = EvalRunner(
                skill_path=skill_path,
                evals_path=evals_path,
                workspace_base=workspace,
                agents=agent_types,
                concurrency=concurrency,
                with_baseline=baseline,
                grader_model=grader["model"],
                grader_base_url=grader["base_url"],
                grader_key_file=grader["key_file"],
                grader_allowed_models=grader["allowed_models"],
                source_repo=source_repo,
                agent_models=agent_model_map,
                harness_base_url=harness_base_url,
                agent_timeout=agent_timeout,
                agent_max_retries=agent_retries,
                runs=runs,
                side_effect_levels=side_effect_levels,
                budget=budget,
                pricing=pricing,
                pre_run_commands=pre_run_commands,
                post_grade_commands=post_grade_commands,
                post_run_commands=post_run_commands,
                reasoning_effort=reasoning_effort,
            )
            result_dir = runner.run(iteration)
        except Exception as e:
            entry = {
                "status": "failed",
                "skill": pair["skill"],
                "skill_dir": pair["skill_dir"],
                "evals_path": pair["evals_path"],
                "error": str(e),
            }
            console.print(f"[red]Run failed for {pair['skill']}: {e}[/red]")
            return {**base, "failures": 1, "entry": entry}

        summary = load_summary(result_dir) or build_summary_from_artifacts(result_dir)
        suite_entry = {
            "status": "ok",
            "skill": pair["skill"],
            "skill_dir": pair["skill_dir"],
            "evals_path": pair["evals_path"],
            "workspace": str(result_dir.parent),
            "iteration_dir": str(result_dir),
        }
        if summary:
            runs_failed_grading = int(summary.get("runs_failed_grading") or 0)
            base["runs_failed_grading"] = runs_failed_grading
            if runs_failed_grading:
                base["suites_failed_grading"] = 1
                suite_entry["status"] = "failed_assertions"
                console.print(f"[yellow]{pair['skill']}: {runs_failed_grading} run(s) had failed assertions[/yellow]")
            suite_entry["summary"] = {
                "runs_completed": summary.get("runs_completed", 0),
                "runs_failed_grading": runs_failed_grading,
                "errors": len(summary.get("errors") or []),
                "skipped_runs": len(summary.get("skipped_runs") or []),
                "duration_seconds": summary.get("duration_seconds", 0),
                "tokens": summary.get("tokens") or {},
                "cost": summary.get("cost") or {},
                "pass_rates": summary.get("pass_rates") or {},
                "failed_runs": summary.get("failed_runs") or [],
            }
            if min_pass_rate is not None:
                below = {
                    config: rate
                    for config, rate in (summary.get("pass_rates") or {}).items()
                    if rate is not None and rate < min_pass_rate
                }
                suite_entry["summary"]["pass_rates_below_threshold"] = below
                if below:
                    base["suites_below_min_pass_rate"] = 1
                    if suite_entry["status"] == "ok":
                        suite_entry["status"] = "below_threshold"
                    console.print(
                        f"[yellow]{pair['skill']}: pass rate below {min_pass_rate:.0%} for "
                        f"{', '.join(f'{config}={rate:.0%}' for config, rate in sorted(below.items()))}[/yellow]"
                    )
        else:
            console.print(f"[yellow]Warning: no readable summary for {pair['skill']} at {result_dir}[/yellow]")

        return {**base, "entry": suite_entry}

    suite_results: list[dict] = []
    if suite_concurrency == 1:
        suite_results = [_suite_result(index, pair) for index, pair in enumerate(pairs)]
    else:
        with ThreadPoolExecutor(max_workers=suite_concurrency) as executor:
            futures = {executor.submit(_suite_result, index, pair): (index, pair) for index, pair in enumerate(pairs)}
            for future in as_completed(futures):
                index, pair = futures[future]
                try:
                    suite_results.append(future.result())
                except Exception as e:
                    console.print(f"[red]Run failed for {pair['skill']}: {e}[/red]")
                    suite_results.append(
                        {
                            "index": index,
                            "failures": 1,
                            "skipped": 0,
                            "runs_failed_grading": 0,
                            "suites_failed_grading": 0,
                            "suites_below_min_pass_rate": 0,
                            "entry": {
                                "status": "failed",
                                "skill": pair["skill"],
                                "skill_dir": pair["skill_dir"],
                                "evals_path": pair["evals_path"],
                                "error": str(e),
                            },
                        }
                    )

    for result in sorted(suite_results, key=lambda item: item["index"]):
        aggregate["failures"] += result["failures"]
        aggregate["skipped"] += result["skipped"]
        aggregate["runs_failed_grading"] += result["runs_failed_grading"]
        aggregate["suites_failed_grading"] += result["suites_failed_grading"]
        aggregate["suites_below_min_pass_rate"] += result["suites_below_min_pass_rate"]
        aggregate["suites"].append(result["entry"])

    manifest_path = workspace / f"run-all-iteration-{iteration}.json"
    aggregate["duration_seconds"] = round(time.time() - started, 1)
    aggregate["clean"] = (
        aggregate["failures"] == 0
        and aggregate["suites_failed_grading"] == 0
        and aggregate["suites_below_min_pass_rate"] == 0
    )
    manifest_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote aggregate manifest: {manifest_path}[/green]")
    if junit_xml:
        junit_path = _write_run_all_junit(aggregate, junit_xml)
        console.print(f"[green]Wrote JUnit XML report: {junit_path}[/green]")
    if aggregate["suites_failed_grading"]:
        console.print(
            f"[yellow]{aggregate['suites_failed_grading']} suite(s) had failed assertions "
            f"across {aggregate['runs_failed_grading']} run(s)[/yellow]"
        )
    if aggregate["failures"]:
        console.print(f"[red]{aggregate['failures']} suite(s) failed[/red]")
        raise typer.Exit(1)
    if fail_on_assertions and aggregate["suites_failed_grading"]:
        raise typer.Exit(1)
    if aggregate["suites_below_min_pass_rate"]:
        raise typer.Exit(1)


@app.command()
def doctor(
    skill: Optional[Path] = typer.Option(None, "--skill", "-s", help="Optional skill directory to check"),
    evals: Optional[Path] = typer.Option(None, "--evals", "-e", help="Optional evals.json file to validate"),
    agents: list[str] = typer.Option(
        ["opencode"],
        "--agent",
        "-a",
        help=f"Agent CLI(s) to check: {AGENT_CHOICES}",
    ),
    eval_ids: list[str] = typer.Option(
        [],
        "--eval-id",
        help="Only validate the eval case with this id. Repeatable.",
    ),
    side_effect_levels: list[str] = typer.Option(
        [],
        "--side-effect-level",
        help=(
            "Only validate eval cases with this side-effect level. "
            f"Repeatable. Choices: {', '.join(SIDE_EFFECT_LEVELS)}"
        ),
    ),
    source_repo: Optional[str] = typer.Option(
        None,
        "--source-repo",
        help="Source repo URL planned for a run; checks gh for GitHub or glab for GitLab",
    ),
    grader_model: str = typer.Option(
        DEFAULT_GRADER_MODEL,
        "--grader-model",
        help="LLM grader model planned for a run; pass '' to disable LLM-grader credential checks",
    ),
    grader_base_url: Optional[str] = typer.Option(None, "--grader-base-url", help="Custom API base URL for grader"),
    grader_key_file: Optional[Path] = typer.Option(
        None,
        "--grader-key-file",
        help="Read the LLM grader API key from this file instead of OPENROUTER_API_KEY/OPENAI_API_KEY",
    ),
    grader_openwebui: bool = typer.Option(
        False,
        "--grader-openwebui",
        help="Use the DT OpenWebUI grader endpoint; only allows Gemini 3.1 Flash Lite",
    ),
):
    """Preflight local tools, skill metadata, eval files, and grader credentials."""

    failures = 0
    warnings = 0
    table = Table(title="agent-skill-eval doctor")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail")

    def add_check(name: str, status: str, detail: str) -> None:
        nonlocal failures, warnings
        if status == "fail":
            failures += 1
            rendered = "[red]fail[/red]"
        elif status == "warn":
            warnings += 1
            rendered = "[yellow]warn[/yellow]"
        else:
            rendered = "[green]ok[/green]"
        table.add_row(name, rendered, detail)

    agent_types = _parse_agent_type_options(agents)
    for agent_type in agent_types:
        command = HARNESSES[agent_type].version_command
        if command is None:
            add_check(f"agent:{agent_type.value}", "ok", "built-in harness; no external CLI required")
            continue
        version = get_cli_version(agent_type)
        if version:
            add_check(f"agent:{agent_type.value}", "ok", version)
        else:
            add_check(
                f"agent:{agent_type.value}",
                "fail",
                f"Could not run {' '.join(command)}; install/authenticate the CLI before running evals",
            )

    if skill is not None:
        try:
            from agent_skill_eval.skills import SkillInstaller

            installer = SkillInstaller(skill)
            problems = installer.frontmatter_problems()
            skill_warnings = installer.frontmatter_warnings()
            if problems:
                add_check("skill", "fail", "; ".join(problems))
            elif skill_warnings:
                add_check("skill", "warn", "; ".join(skill_warnings))
            else:
                add_check("skill", "ok", str(skill))
        except FileNotFoundError as e:
            add_check("skill", "fail", str(e))
    else:
        add_check("skill", "warn", "No --skill supplied; skipped skill metadata checks")

    report: EvalSuiteValidationReport | None = None
    if evals is not None:
        try:
            _, report = load_and_validate_suite(evals, eval_ids=eval_ids, side_effect_levels=side_effect_levels)
            detail = f"{report.selected_eval_count} eval(s), {report.total_assertions} assertion(s)"
            add_check("evals", "ok", detail)
        except EvalSuiteValidationError as e:
            add_check("evals", "fail", str(e).replace("\n", "; "))
    else:
        add_check("evals", "warn", "No --evals supplied; skipped suite validation")

    grader = _grader_settings(
        grader_model=grader_model,
        grader_base_url=grader_base_url,
        grader_key_file=grader_key_file,
        grader_openwebui=grader_openwebui,
    )
    grader_has_credentials = LLMGrader.has_credentials(grader["key_file"]) if grader["model"] else False
    if grader["model"] and report and report.llm_assertions and not grader_has_credentials:
        missing = _missing_grader_credentials_message(grader["key_file"])
        add_check(
            "llm-grader",
            "fail",
            f"{len(report.llm_assertions)} assertion(s) need LLM grading, but {missing}",
        )
    elif grader["model"] and not grader_has_credentials:
        add_check("llm-grader", "warn", "No grader API key available; deterministic-only suites can still run")
    elif grader["model"]:
        backend = "OpenWebUI" if grader["openwebui"] else "OpenAI-compatible"
        add_check("llm-grader", "ok", f"{backend} credentials available for {grader['model']}")
    else:
        add_check("llm-grader", "ok", "disabled by --grader-model ''")

    source_provider = repo_provider(source_repo)
    needs_glab = bool(source_repo and source_provider == "gitlab")
    needs_gh = bool(source_repo and source_provider != "gitlab")
    if report:
        needs_gh = needs_gh or (
            source_provider != "gitlab"
            and any(item.method == "pr-created" for item in report.assertion_classifications)
        )
    if needs_gh:
        gh = shutil.which("gh")
        add_check("gh", "ok" if gh else "fail", gh or "gh CLI not found on PATH")
    else:
        add_check("gh", "ok", "not required by supplied options/evals")
    if needs_glab:
        glab = shutil.which("glab")
        add_check("glab", "ok" if glab else "fail", glab or "glab CLI not found on PATH")
    else:
        add_check("glab", "ok", "not required by supplied options")

    console.print(table)
    if failures:
        console.print(f"[red]Doctor found {failures} failure(s) and {warnings} warning(s).[/red]")
        raise typer.Exit(1)
    if warnings:
        console.print(f"[yellow]Doctor found {warnings} warning(s).[/yellow]")
    else:
        console.print("[green]Doctor checks passed.[/green]")


def _parse_agent_type_options(agents: list[str]) -> list[AgentType]:
    agent_types = []
    for agent in agents:
        try:
            agent_types.append(AgentType(agent))
        except ValueError:
            console.print(f"[red]Unknown agent: {agent}. Choose from: {AGENT_CHOICES}[/red]")
            raise typer.Exit(1)
    return agent_types


def _parse_agent_models(specs: list[str], agent_types: list[AgentType]) -> dict[AgentType, str]:
    """Parse repeated ``--agent-model`` values into an AgentType -> model map.

    Accepts ``agent=model`` to target one agent, or a bare ``model`` that
    applies to every selected agent.
    """
    mapping: dict[AgentType, str] = {}
    for spec in specs:
        if "=" in spec:
            agent_str, model = spec.split("=", 1)
            try:
                agent = AgentType(agent_str.strip())
            except ValueError:
                console.print(f"[red]Unknown agent in --agent-model '{spec}'. Choose from: {AGENT_CHOICES}[/red]")
                raise typer.Exit(1)
            if not model.strip():
                console.print(f"[red]Empty model in --agent-model '{spec}'[/red]")
                raise typer.Exit(1)
            mapping[agent] = model.strip()
        else:
            if not spec.strip():
                console.print(f"[red]Empty model in --agent-model '{spec}'[/red]")
                raise typer.Exit(1)
            for agent in agent_types:
                mapping.setdefault(agent, spec.strip())
    return mapping


def _load_pricing_config(path: Optional[Path]) -> dict[str, dict]:
    """Load a ``--pricing-config`` file: ``{model_id: {rate_field: usd_per_token}}``.

    Rate fields mirror OpenRouter's pricing schema: ``prompt``, ``completion``,
    ``input_cache_read``, ``input_cache_write``.
    """
    if path is None:
        return {}
    if not path.exists():
        console.print(f"[red]Pricing config not found: {path}[/red]")
        raise typer.Exit(1)
    if not path.is_file():
        console.print(f"[red]Pricing config {path} is not a file[/red]")
        raise typer.Exit(1)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in pricing config {path}: {e}[/red]")
        raise typer.Exit(1)
    if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
        console.print(f"[red]Pricing config {path} must map model ids to rate objects[/red]")
        raise typer.Exit(1)
    return data


def _cleanup_iteration(iteration_dir: Path, workspace_base: Path) -> None:
    """Clean up only artifacts recorded for this iteration's cleanup manifest.

    Reads ``cleanup.json`` from the iteration dir. If the manifest is missing,
    we do NOT touch remote state (no "delete all branches" or "close all review
    requests" inference). We also do NOT glob-delete unrecorded ``agent-skill-eval-*``
    workspaces under the workspace base, because those may belong to other
    runs or manual debugging. The only workspaces removed are those listed
    in the manifest.
    """
    manifest = _load_manifest(iteration_dir)

    merge_requests = manifest.merge_request_numbers if manifest else []
    pull_requests = manifest.pr_numbers if manifest else []
    if manifest and (manifest.remote_branches or pull_requests or merge_requests):
        _cleanup_manifest(manifest, source_repo=manifest.source_repo)
    elif manifest is None and iteration_dir.exists():
        console.print(
            f"  [yellow]No cleanup.json in {iteration_dir}; skipping source repo cleanup "
            f"and unrecorded workspace deletion[/yellow]"
        )

    if manifest:
        for ws_path in manifest.workspaces:
            ws = Path(ws_path)
            if ws.exists():
                shutil.rmtree(ws)
                console.print(f"  [dim]Removed {ws.name}[/dim]")


def _load_manifest(iteration_dir: Path) -> Optional[CleanupManifest]:
    manifest_path = iteration_dir / "cleanup.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text())
        return CleanupManifest(**data)
    except (json.JSONDecodeError, ValueError) as e:
        console.print(f"  [red]Could not parse cleanup.json: {e}[/red]")
        return None


def _cleanup_manifest(manifest: CleanupManifest, source_repo: Optional[str]) -> None:
    if not source_repo:
        return

    provider = manifest.source_repo_provider or ("github" if manifest.source_repo_slug else None)
    project = manifest.source_repo_project or manifest.source_repo_slug
    if not project:
        return

    for pr_number in manifest.pr_numbers:
        if provider == "github":
            _close_github_pr(pr_number, project)
        elif provider == "gitlab":
            _close_gitlab_mr(pr_number, project)
        else:
            console.print(f"  [yellow]Unknown provider; cannot close review #{pr_number}[/yellow]")

    for mr_number in manifest.merge_request_numbers:
        if mr_number in manifest.pr_numbers:
            continue
        if provider == "gitlab":
            _close_gitlab_mr(mr_number, project)
        else:
            console.print(f"  [yellow]Merge request cleanup is only supported for GitLab: !{mr_number}[/yellow]")

    for branch in manifest.remote_branches:
        result = subprocess.run(
            ["git", "push", source_repo, "--delete", branch],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            console.print(f"  [dim]Deleted branch {branch}[/dim]")
        else:
            console.print(
                f"  [yellow]Could not delete branch {branch}: {result.stderr.strip() or 'unknown error'}[/yellow]"
            )


def _close_github_pr(pr_number: int, project: str) -> None:
    result = subprocess.run(
        ["gh", "pr", "close", str(pr_number), "--repo", project],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        console.print(f"  [dim]Closed PR #{pr_number}[/dim]")
    else:
        console.print(f"  [yellow]Could not close PR #{pr_number}: {result.stderr.strip() or 'unknown error'}[/yellow]")


def _close_gitlab_mr(mr_number: int, project: str) -> None:
    result = subprocess.run(
        ["glab", "mr", "close", str(mr_number), "-R", project],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        console.print(f"  [dim]Closed MR !{mr_number}[/dim]")
    else:
        console.print(f"  [yellow]Could not close MR !{mr_number}: {result.stderr.strip() or 'unknown error'}[/yellow]")


@app.command()
def cleanup(
    iteration: Optional[Path] = typer.Option(
        None,
        "--iteration",
        "-i",
        help="Path to an iteration directory (e.g. eval-workspace/<skill>-workspace/iteration-1). "
        "If omitted, all iteration dirs under --workspace are cleaned up.",
    ),
    workspace: Path = typer.Option(
        Path.cwd() / "eval-workspace",
        "--workspace",
        "-w",
        help="Base directory for eval workspace (used when --iteration is omitted)",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clean up eval artifacts from cleanup.json manifests.

    This command is SAFE: it only closes PRs/MRs and deletes branches that were
    recorded in ``cleanup.json`` by a previous ``run`` invocation. It will
    never close unrelated review requests or delete unrelated branches.
    """
    iteration_dirs: list[Path] = []
    if iteration:
        if not iteration.exists():
            console.print(f"[red]Iteration directory not found: {iteration}[/red]")
            raise typer.Exit(1)
        iteration_dirs = [iteration]
    else:
        if not workspace.exists():
            console.print(f"[yellow]Workspace directory not found: {workspace}[/yellow]")
            return
        iteration_dirs = sorted(p for p in workspace.glob("*/iteration-*") if p.is_dir())

    if not iteration_dirs:
        console.print("[yellow]No iteration directories found[/yellow]")
        return

    for iter_dir in iteration_dirs:
        manifest = _load_manifest(iter_dir)
        if manifest is None:
            console.print(f"[yellow]{iter_dir.name}: no cleanup.json, skipping source repo cleanup[/yellow]")
        elif manifest.remote_branches or manifest.pr_numbers or manifest.merge_request_numbers:
            review_count = len(manifest.merge_request_numbers or manifest.pr_numbers)
            project = manifest.source_repo_project or manifest.source_repo_slug
            console.print(
                f"[cyan]{iter_dir.name}: cleaning {len(manifest.remote_branches)} branch(es), "
                f"{review_count} review request(s) on {project}[/cyan]"
            )

            if not yes:
                confirm = typer.confirm("Proceed?", default=False)
                if not confirm:
                    console.print("  [dim]Skipped[/dim]")
                    continue

            _cleanup_manifest(manifest, source_repo=manifest.source_repo)

        if manifest and manifest.workspaces:
            for ws_path in manifest.workspaces:
                ws = Path(ws_path)
                if ws.exists():
                    shutil.rmtree(ws)

    console.print("[green]Cleanup complete![/green]")


@app.command()
def grade(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Path to iteration workspace"),
    grader_model: str = typer.Option(DEFAULT_GRADER_MODEL, "--grader-model", help="LLM model for rubric grading"),
    grader_base_url: Optional[str] = typer.Option(None, "--grader-base-url", help="Custom API base URL for grader"),
    grader_key_file: Optional[Path] = typer.Option(
        None,
        "--grader-key-file",
        help="Read the LLM grader API key from this file instead of OPENROUTER_API_KEY/OPENAI_API_KEY",
    ),
    grader_openwebui: bool = typer.Option(
        False,
        "--grader-openwebui",
        help="Use the DT OpenWebUI grader endpoint; only allows Gemini 3.1 Flash Lite",
    ),
    recompute_benchmark: bool = typer.Option(
        False, "--recompute-benchmark", help="Recompute benchmark.json from updated grading.json files"
    ),
):
    """Re-grade existing eval results using saved eval metadata and state snapshots."""
    if not workspace.exists():
        console.print(f"[red]Workspace not found: {workspace}[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]Re-grading results in {workspace}...[/yellow]")
    console.print(
        "[yellow]Note: LLM-graded assertions are re-evaluated from scratch; "
        "their verdicts may differ from the original run.[/yellow]"
    )

    meta_path = workspace / "evals_meta.json"
    if not meta_path.exists():
        console.print(
            f"[red]No evals_meta.json in {workspace}. Cannot determine assertions. "
            f"Re-run 'agent-skill-eval run' to generate it.[/red]"
        )
        raise typer.Exit(1)

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    eval_lookup: dict[str, dict] = {}
    for eval_case in meta.get("evals", []):
        eval_lookup[str(eval_case["id"])] = eval_case

    grader = _grader_settings(
        grader_model=grader_model,
        grader_base_url=grader_base_url,
        grader_key_file=grader_key_file,
        grader_openwebui=grader_openwebui,
    )
    try:
        llm_grader = (
            LLMGrader(
                model=grader["model"],
                base_url=grader["base_url"],
                api_key_file=grader["key_file"],
                allowed_models=grader["allowed_models"],
            )
            if grader["model"]
            else None
        )
    except Exception:
        llm_grader = None

    updated = 0
    for eval_dir in sorted(workspace.glob("eval-*")):
        eval_id = eval_dir.name[len("eval-") :]
        eval_case = eval_lookup.get(eval_id)
        if not eval_case:
            console.print(f"  [yellow]No metadata for {eval_dir.name}, skipping[/yellow]")
            continue

        assertions = eval_case.get("assertions", [])
        should_trigger = eval_case.get("should_trigger", True)

        for agent_dir in sorted(eval_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            for config_dir in _leaf_config_dirs(agent_dir):
                output_path = config_dir / "outputs" / "output.txt"
                if not output_path.exists():
                    continue

                agent_output = output_path.read_text()
                grading = grade_assertions(
                    assertions,
                    agent_output,
                    config_dir / "outputs",
                    eval_case.get("expected_output", ""),
                    llm_grader,
                    pre_state=None,
                    post_state=None,
                    should_trigger=should_trigger,
                )

                grading_path = config_dir / "grading.json"
                with open(grading_path, "w", encoding="utf-8") as f:
                    json.dump(grading.model_dump(), f, indent=2)

                console.print(
                    f"  {eval_dir.name}/{agent_dir.name}/{config_dir.relative_to(agent_dir)}: "
                    f"{grading.summary.passed}/{grading.summary.total} passed"
                )
                updated += 1

    if recompute_benchmark:
        from agent_skill_eval.runner import compute_benchmark

        results = _collect_results(workspace)
        agents = sorted({r.get("agent") for r in results.values() if r.get("agent") and r.get("agent") != "unknown"})
        agent_types = [AgentType(a) for a in agents if a in {t.value for t in AgentType}]
        benchmark = compute_benchmark(
            results, agent_types, with_baseline=any(not r.get("with_skill", True) for r in results.values())
        )
        with open(workspace / "benchmark.json", "w", encoding="utf-8") as f:
            json.dump(benchmark.model_dump(), f, indent=2)
        console.print("[green]benchmark.json recomputed[/green]")
    else:
        console.print("[yellow]benchmark.json NOT recomputed. Re-run 'ase run' or pass --recompute-benchmark.[/yellow]")

    console.print(f"[green]Re-grading complete! Updated {updated} grading file(s).[/green]")


def _collect_results(workspace: Path) -> dict[str, dict]:
    """Reconstruct per-run results for benchmark recompute.

    Reads ``run_meta.json`` written by ``_run_single`` to recover the real
    agent and ``with_skill`` flag (the directory layout now includes the
    agent, but we use the metadata file as the source of truth). Falls back
    to ``"unknown"`` only if the metadata is missing, and disambiguates
    collisions by including the agent directory name in the key.
    """
    results: dict[str, dict] = {}
    for eval_dir in sorted(workspace.glob("eval-*")):
        for agent_dir in sorted(eval_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            for config_dir in _leaf_config_dirs(agent_dir):
                grading_path = config_dir / "grading.json"
                timing_path = config_dir / "timing.json"
                if not grading_path.exists() or not timing_path.exists():
                    continue
                with open(grading_path, encoding="utf-8") as f:
                    grading = json.load(f)
                with open(timing_path, encoding="utf-8") as f:
                    timing = json.load(f)

                meta_path = config_dir / "run_meta.json"
                agent = "unknown"
                with_skill = "with_skill" in config_dir.parts
                eval_id = eval_dir.name[len("eval-") :]
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                        agent = meta.get("agent", "unknown")
                        with_skill = bool(meta.get("with_skill", with_skill))
                        eval_id = meta.get("eval_id", eval_id)
                        side_effect_level = meta.get("side_effect_level", "local-only")
                        skill_hash = meta.get("skill_hash")
                    except (json.JSONDecodeError, ValueError):
                        console.print(f"  [yellow]Could not parse {meta_path}; marking agent as unknown[/yellow]")
                        side_effect_level = "local-only"
                        skill_hash = None
                else:
                    console.print(f"  [yellow]No run_meta.json in {config_dir}; agent identity unknown[/yellow]")
                    side_effect_level = "local-only"
                    skill_hash = None

                # Include the path relative to the eval dir in the key to
                # avoid collisions when metadata is missing.
                key = f"{eval_dir.name}-{agent}-{'-'.join(config_dir.relative_to(eval_dir).parts)}"
                if key in results:
                    console.print(f"  [yellow]Duplicate result key {key} for {config_dir}; skipping[/yellow]")
                    continue
                results[key] = {
                    "eval_id": eval_id,
                    "agent": agent,
                    "with_skill": with_skill,
                    "side_effect_level": side_effect_level,
                    "skill_hash": skill_hash,
                    "timing": timing,
                    "grading": grading,
                }
    return results


def _benchmark_rows(benchmark: dict) -> list[tuple[str, ...]]:
    rows = []
    for config, stats in benchmark.get("run_summary", {}).items():
        pass_rate = stats.get("pass_rate", {})
        time_s = stats.get("time_seconds", {})
        tokens = stats.get("tokens", {})
        # None/absent means cost is unavailable for these runs (e.g. codex
        # without --pricing-config), which is different from a real $0.
        cost = stats.get("cost_usd")
        cost_str = f"{cost.get('median', 0):.4f} / {cost.get('p90', 0):.4f}" if cost else "n/a"
        hat = stats.get("pass_hat_k", stats.get("full_pass_rate", 0.0))
        low = stats.get("pass_hat_k_wilson_low", 0.0)
        high = stats.get("pass_hat_k_wilson_high", 1.0)
        pass_hat = f"{hat:.0%} [{low:.0%}-{high:.0%}] k={stats.get('k', 1)}"
        rows.append(
            (
                config,
                f"{pass_rate.get('mean', 0):.1%} +/- {pass_rate.get('stddev', 0):.1%}",
                pass_hat,
                f"{time_s.get('median', 0):.1f} / {time_s.get('p90', 0):.1f}",
                f"{tokens.get('median', 0):.0f} / {tokens.get('p90', 0):.0f}",
                str(stats.get("skipped_assertions", 0)),
                cost_str,
            )
        )
    return rows


def _analysis_lines(summary: dict) -> list[str]:
    lines: list[str] = []
    for config, trigger in (summary.get("trigger_rate") or {}).items():
        positive = trigger.get("positive") or {}
        negative = trigger.get("negative") or {}
        lines.append(
            f"`{config}` trigger: positive {positive.get('rate')} "
            f"({positive.get('invoked')}/{positive.get('known')} known, "
            f"{positive.get('unknown')} unknown); negative {negative.get('rate')} "
            f"({negative.get('invoked')}/{negative.get('known')} known, "
            f"{negative.get('unknown')} unknown)"
        )
    for config, tags in (summary.get("tag_pass_rates") or {}).items():
        rendered = ", ".join(f"{tag} {rate:.1%}" for tag, rate in tags.items())
        lines.append(f"`{config}` pass by tag: {rendered}")
    for config, access in (summary.get("out_of_workspace") or {}).items():
        examples = ", ".join((access.get("examples") or [])[:3])
        lines.append(
            f"`{config}` out-of-workspace: {access.get('runs_with_access')}/"
            f"{access.get('total')} runs" + (f"; e.g. {examples}" if examples else "")
        )
    return lines


def _delta_line(delta: dict) -> str:
    cost = delta.get("cost_usd")
    cost_str = f"{cost:+.4f} USD" if cost is not None else "n/a"
    return (
        f"pass rate {delta.get('pass_rate', 0):+.1%}, "
        f"time {delta.get('time_seconds', 0):+.1f}s, tokens {delta.get('tokens', 0):+.0f}, "
        f"cost {cost_str}"
    )


def _benchmark_deltas(benchmark: dict) -> dict[str, dict]:
    deltas = benchmark.get("deltas") or {}
    if not deltas and benchmark.get("delta"):
        deltas = {"all": benchmark["delta"]}
    return deltas


def _metric_mean(benchmark: dict, config: str, field: str):
    return ((benchmark.get("run_summary") or {}).get(config, {}).get(field) or {}).get("mean")


def _config_stats(benchmark: dict, config: str) -> dict:
    return (benchmark.get("run_summary") or {}).get(config, {})


def _format_cost_delta(a_stats: dict, b_stats: dict) -> str:
    a = (a_stats.get("cost_usd") or {}).get("mean")
    b = (b_stats.get("cost_usd") or {}).get("mean")
    if a is None or b is None:
        return "-"
    return f"{b - a:+.4f}"


def _failed_assertion_counter(summary: dict) -> Counter[str]:
    counter: Counter[str] = Counter()
    for run in summary.get("failed_runs") or []:
        for assertion in run.get("failed_assertions") or []:
            counter[str(assertion)] += 1
    return counter


def _skill_file_index(summary: dict) -> dict[str, str]:
    return {
        item.get("path", ""): item.get("sha256", "")
        for item in ((summary.get("skill_size") or {}).get("content_files") or [])
        if item.get("path")
    }


def _changed_skill_files(summary_a: dict, summary_b: dict) -> list[str]:
    files_a = _skill_file_index(summary_a)
    files_b = _skill_file_index(summary_b)
    changed = []
    for path in sorted(set(files_a) | set(files_b)):
        if files_a.get(path) != files_b.get(path):
            changed.append(path)
    return changed


def _compare_rows(benchmark_a: dict, benchmark_b: dict) -> list[dict]:
    configs = sorted(set((benchmark_a.get("run_summary") or {})) | set((benchmark_b.get("run_summary") or {})))
    rows: list[dict] = []
    for config in configs:
        stats_a = _config_stats(benchmark_a, config)
        stats_b = _config_stats(benchmark_b, config)
        pass_a = _metric_mean(benchmark_a, config, "pass_rate")
        pass_b = _metric_mean(benchmark_b, config, "pass_rate")
        time_a = _metric_mean(benchmark_a, config, "time_seconds")
        time_b = _metric_mean(benchmark_b, config, "time_seconds")
        tokens_a = _metric_mean(benchmark_a, config, "tokens")
        tokens_b = _metric_mean(benchmark_b, config, "tokens")
        rows.append(
            {
                "configuration": config,
                "pass_rate_a": pass_a,
                "pass_rate_b": pass_b,
                "pass_rate_delta": (pass_b - pass_a) if pass_a is not None and pass_b is not None else None,
                "full_pass_a": stats_a.get("full_pass_rate"),
                "full_pass_b": stats_b.get("full_pass_rate"),
                "pass_at_k_a": stats_a.get("pass_at_k"),
                "pass_at_k_b": stats_b.get("pass_at_k"),
                "time_delta": (time_b - time_a) if time_a is not None and time_b is not None else None,
                "tokens_delta": (tokens_b - tokens_a) if tokens_a is not None and tokens_b is not None else None,
                "cost_delta": _format_cost_delta(stats_a, stats_b),
            }
        )
    return rows


REPORT_COLUMNS = [
    "Configuration",
    "Pass Rate",
    "pass^k [95% CI]",
    "Time med/p90 (s)",
    "Tokens med/p90",
    "Skipped",
    "Cost med/p90 (USD)",
]


def _print_markdown_report(iter_name: str, benchmark: dict, summary: dict) -> None:
    print(f"\n### {iter_name}\n")
    print("| " + " | ".join(REPORT_COLUMNS) + " |")
    print("|" + "|".join([" --- "] * len(REPORT_COLUMNS)) + "|")
    for row in _benchmark_rows(benchmark):
        print("| " + " | ".join(row) + " |")
    deltas = _benchmark_deltas(benchmark)
    if deltas:
        print("\n**Delta (with_skill - without_skill):**\n")
        for agent, delta in deltas.items():
            print(f"- `{agent}`: {_delta_line(delta)}")
    analysis = _analysis_lines(summary)
    if analysis:
        print("\n**Skill analysis:**\n")
        for line in analysis:
            print(f"- {line}")


def _html(value) -> str:
    return html.escape(str(value), quote=True)


def _print_html_report(iter_name: str, benchmark: dict, summary: dict) -> None:
    print(f'<section class="iteration"><h2>{_html(iter_name)}</h2>')
    print("<table><thead><tr>")
    for column in REPORT_COLUMNS:
        print(f"<th>{_html(column)}</th>")
    print("</tr></thead><tbody>")
    for row in _benchmark_rows(benchmark):
        print("<tr>" + "".join(f"<td>{_html(cell)}</td>" for cell in row) + "</tr>")
    print("</tbody></table>")
    deltas = _benchmark_deltas(benchmark)
    if deltas:
        print("<h3>Delta (with_skill - without_skill)</h3><ul>")
        for agent, delta in deltas.items():
            print(f"<li><code>{_html(agent)}</code>: {_html(_delta_line(delta))}</li>")
        print("</ul>")
    analysis = _analysis_lines(summary)
    if analysis:
        print("<h3>Skill analysis</h3><ul>")
        for line in analysis:
            print(f"<li>{_html(line)}</li>")
        print("</ul>")


def _print_table_report(iter_name: str, benchmark: dict, summary: dict) -> None:
    console.print(f"\n[bold]{iter_name}[/bold]")

    table = Table(title="Results Summary")
    styles = ["cyan", "green", "magenta", "yellow", "blue", "white"]
    for column, style in zip(REPORT_COLUMNS, styles):
        table.add_column(column, style=style)
    for row in _benchmark_rows(benchmark):
        table.add_row(*row)
    console.print(table)

    deltas = _benchmark_deltas(benchmark)
    if deltas:
        console.print("\n[bold]Delta (with_skill - without_skill):[/bold]")
        for agent, delta in deltas.items():
            console.print(f"  {agent}: {_delta_line(delta)}")
    analysis = _analysis_lines(summary)
    if analysis:
        console.print("\n[bold]Skill analysis:[/bold]")
        for line in analysis:
            console.print(f"  {line}")


def _print_eval_details(iter_dir: Path, show_evidence: bool, markdown: bool, failures_only: bool = False) -> None:
    for eval_dir in sorted(iter_dir.glob("eval-*")):
        header_printed = False
        for agent_dir in sorted(eval_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            for config_dir in _leaf_config_dirs(agent_dir):
                grading_path = config_dir / "grading.json"
                if not grading_path.exists():
                    continue
                with open(grading_path, encoding="utf-8") as f:
                    grading = json.load(f)
                summary = grading.get("summary", {})
                if failures_only and summary.get("failed", 0) == 0:
                    continue
                if not header_printed:
                    if markdown:
                        print(f"\n#### {eval_dir.name}\n")
                    else:
                        console.print(f"\n[bold]{eval_dir.name}[/bold]")
                    header_printed = True
                label = f"{agent_dir.name}/{config_dir.relative_to(agent_dir)}"
                skipped = summary.get("skipped", 0)
                skipped_note = f", {skipped} skipped" if skipped else ""
                line = (
                    f"{label}: {summary.get('passed', 0)}/{summary.get('total', 0)} passed "
                    f"({summary.get('pass_rate', 0):.0%}{skipped_note})"
                )
                if markdown:
                    print(f"- {line}")
                else:
                    console.print(f"  {line}")
                if show_evidence:
                    for result in grading.get("assertion_results", []):
                        if result.get("passed"):
                            continue
                        status = "SKIPPED" if result.get("skipped") else "FAIL"
                        detail = f"[{status}] {result.get('text')} — {result.get('evidence')}"
                        if markdown:
                            print(f"  - {detail}")
                        else:
                            style = "yellow" if result.get("skipped") else "red"
                            console.print(f"    [{style}]{detail}[/{style}]")


def _print_html_eval_details(iter_dir: Path, show_evidence: bool, failures_only: bool = False) -> None:
    for eval_dir in sorted(iter_dir.glob("eval-*")):
        runs: list[str] = []
        for agent_dir in sorted(eval_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            for config_dir in _leaf_config_dirs(agent_dir):
                grading_path = config_dir / "grading.json"
                if not grading_path.exists():
                    continue
                with open(grading_path, encoding="utf-8") as f:
                    grading = json.load(f)
                summary = grading.get("summary", {})
                if failures_only and summary.get("failed", 0) == 0:
                    continue
                label = f"{agent_dir.name}/{config_dir.relative_to(agent_dir)}"
                skipped = summary.get("skipped", 0)
                skipped_note = f", {skipped} skipped" if skipped else ""
                line = (
                    f"{label}: {summary.get('passed', 0)}/{summary.get('total', 0)} passed "
                    f"({summary.get('pass_rate', 0):.0%}{skipped_note})"
                )
                details = ""
                if show_evidence:
                    failed = [result for result in grading.get("assertion_results", []) if not result.get("passed")]
                    if failed:
                        items = "".join(
                            "<li>"
                            f"<strong>{'SKIPPED' if result.get('skipped') else 'FAIL'}</strong> "
                            f"{_html(result.get('text'))}: {_html(result.get('evidence'))}"
                            "</li>"
                            for result in failed
                        )
                        details = f"<ul>{items}</ul>"
                runs.append(f"<li>{_html(line)}{details}</li>")
        if runs:
            print(f"<h3>{_html(eval_dir.name)}</h3><ul>{''.join(runs)}</ul>")
    print("</section>")


def _run_all_manifest_iteration(path: Path) -> int:
    prefix = "run-all-iteration-"
    if not path.name.startswith(prefix) or path.suffix != ".json":
        return -1
    try:
        return int(path.stem[len(prefix) :])
    except ValueError:
        return -1


def _load_run_all_manifest(workspace: Path, iteration: Optional[int]) -> tuple[Path, dict]:
    if iteration is not None:
        manifest_path = workspace / f"run-all-iteration-{iteration}.json"
        if not manifest_path.exists():
            console.print(f"[red]No run-all manifest found: {manifest_path}[/red]")
            raise typer.Exit(1)
    else:
        candidates = sorted(
            (path for path in workspace.glob("run-all-iteration-*.json") if _run_all_manifest_iteration(path) >= 0),
            key=_run_all_manifest_iteration,
        )
        if not candidates:
            console.print(f"[red]No run-all manifests found under {workspace}[/red]")
            raise typer.Exit(1)
        manifest_path = candidates[-1]

    try:
        return manifest_path, json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"[red]Could not read {manifest_path}: {e}[/red]")
        raise typer.Exit(1) from e


def _format_pass_rate(rate) -> str:
    if rate is None:
        return "n/a"
    try:
        return f"{float(rate):.0%}"
    except (TypeError, ValueError):
        return str(rate)


def _format_pass_rates(pass_rates: dict) -> str:
    if not pass_rates:
        return "n/a"
    return ", ".join(f"{config}={_format_pass_rate(rate)}" for config, rate in sorted(pass_rates.items()))


def _format_duration_seconds(value) -> str:
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "n/a"


def _format_token_total(tokens: dict) -> str:
    total = (tokens or {}).get("total")
    if total is None:
        return "n/a"
    try:
        return f"{int(total):,}"
    except (TypeError, ValueError):
        return str(total)


def _format_cost_total(cost: dict) -> str:
    cost = cost or {}
    total = cost.get("total_usd")
    if total is None:
        if cost.get("runs_total"):
            return "unavailable"
        return "n/a"
    try:
        return f"${float(total):.4f}"
    except (TypeError, ValueError):
        return str(total)


def _run_all_rows(manifest: dict, failures_only: bool = False) -> list[dict]:
    rows: list[dict] = []
    for suite in manifest.get("suites", []):
        summary = suite.get("summary") or {}
        status = suite.get("status", "unknown")
        failed = summary.get("runs_failed_grading", 0) or 0
        errors = summary.get("errors", 0) or (1 if suite.get("error") else 0)
        if failures_only and status in ("ok", "skipped") and not failed and not errors:
            continue
        rows.append(
            {
                "skill": suite.get("skill", ""),
                "status": status,
                "cases": summary.get("runs_completed", suite.get("planned_cases", 0)),
                "completed": summary.get("runs_completed", 0),
                "failed": failed,
                "errors": errors,
                "duration": _format_duration_seconds(summary.get("duration_seconds")),
                "tokens": _format_token_total(summary.get("tokens") or {}),
                "cost": _format_cost_total(summary.get("cost") or {}),
                "pass_rates": _format_pass_rates(summary.get("pass_rates") or {}),
                "iteration_dir": suite.get("iteration_dir", ""),
                "reason": suite.get("reason") or suite.get("error") or "",
                "failed_runs": summary.get("failed_runs") or [],
            }
        )
    return rows


def _run_all_summary_line(manifest: dict) -> str:
    clean = manifest.get("clean")
    clean_text = "yes" if clean is True else "no" if clean is False else "unknown"
    plan_text = ""
    if manifest.get("plan_only"):
        plan_text = (
            f", planned_suites={manifest.get('planned_suites', 0)}, planned_cases={manifest.get('planned_cases', 0)}"
        )
    return (
        f"iteration {manifest.get('iteration', 'unknown')}: clean={clean_text}, "
        f"suites={len(manifest.get('suites', []))}, skipped={manifest.get('skipped', 0)}, "
        f"suite_failures={manifest.get('failures', 0)}, "
        f"assertion_failed_suites={manifest.get('suites_failed_grading', 0)}, "
        f"assertion_failed_runs={manifest.get('runs_failed_grading', 0)}, "
        f"duration={_format_duration_seconds(manifest.get('duration_seconds'))}"
        f"{plan_text}"
    )


def _print_run_all_failures(rows: list[dict], *, markdown: bool) -> None:
    for row in rows:
        failed_runs = row.get("failed_runs") or []
        if not failed_runs and not row.get("reason"):
            continue
        if markdown:
            print(f"\n#### {row['skill']}\n")
            if row.get("reason"):
                print(f"- {row['reason']}")
        else:
            console.print(f"\n[bold]{row['skill']}[/bold]")
            if row.get("reason"):
                console.print(f"  {row['reason']}")
        for run in failed_runs:
            label = (
                f"{run.get('eval_id')} / {run.get('agent')} / "
                f"{'with_skill' if run.get('with_skill') else 'without_skill'}"
            )
            pass_rate = _format_pass_rate(run.get("pass_rate"))
            if markdown:
                print(f"- `{label}`: {pass_rate}")
                for assertion in run.get("failed_assertions") or []:
                    print(f"  - {assertion}")
            else:
                console.print(f"  {label}: {pass_rate}")
                for assertion in run.get("failed_assertions") or []:
                    console.print(f"    [red]{assertion}[/red]")


def _print_run_all_html(manifest_path: Path, manifest: dict, rows: list[dict]) -> None:
    print(
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>agent-skill-eval run-all report</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.4}"
        "table{border-collapse:collapse;margin:1rem 0;width:100%}"
        "th,td{border:1px solid #ddd;padding:.45rem;text-align:left;vertical-align:top}"
        "th{background:#f6f8fa}code{background:#f6f8fa;padding:.1rem .25rem;border-radius:4px}"
        "</style></head><body>"
    )
    print(f"<h1>agent-skill-eval run-all report</h1><p>{_html(manifest_path)}</p>")
    print(f"<p>{_html(_run_all_summary_line(manifest))}</p>")
    print("<table><thead><tr>")
    columns = [
        "Skill",
        "Status",
        "Cases",
        "Completed",
        "Failed Runs",
        "Errors",
        "Duration",
        "Tokens",
        "Cost",
        "Pass Rates",
    ]
    for column in columns:
        print(f"<th>{_html(column)}</th>")
    print("</tr></thead><tbody>")
    for row in rows:
        cells = [
            row["skill"],
            row["status"],
            row["cases"],
            row["completed"],
            row["failed"],
            row["errors"],
            row["duration"],
            row["tokens"],
            row["cost"],
            row["pass_rates"],
        ]
        print("<tr>" + "".join(f"<td>{_html(cell)}</td>" for cell in cells) + "</tr>")
    print("</tbody></table>")
    for row in rows:
        failed_runs = row.get("failed_runs") or []
        if not failed_runs and not row.get("reason"):
            continue
        print(f"<h2>{_html(row['skill'])}</h2>")
        if row.get("reason"):
            print(f"<p>{_html(row['reason'])}</p>")
        if failed_runs:
            print("<ul>")
            for run in failed_runs:
                label = (
                    f"{run.get('eval_id')} / {run.get('agent')} / "
                    f"{'with_skill' if run.get('with_skill') else 'without_skill'}"
                )
                assertions = "".join(f"<li>{_html(a)}</li>" for a in run.get("failed_assertions") or [])
                print(f"<li><code>{_html(label)}</code>: {_html(_format_pass_rate(run.get('pass_rate')))}")
                if assertions:
                    print(f"<ul>{assertions}</ul>")
                print("</li>")
            print("</ul>")
    print("</body></html>")


def _load_run_all_manifest_file(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        console.print(f"[red]Run-all manifest not found: {manifest_path}[/red]")
        raise typer.Exit(1)
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"[red]Could not read {manifest_path}: {e}[/red]")
        raise typer.Exit(1) from e


def _shell_command(args: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in args)


def _append_manifest_rerun_context(args: list[str], manifest: dict, agent: str) -> None:
    agent_models = manifest.get("agent_models") or {}
    model = agent_models.get(agent)
    if model:
        args.extend(["--agent-model", f"{agent}={model}"])
    if manifest.get("reasoning_effort"):
        args.extend(["--reasoning-effort", str(manifest["reasoning_effort"])])
    if manifest.get("harness_base_url"):
        args.extend(["--harness-base-url", str(manifest["harness_base_url"])])
    if manifest.get("timeout") is not None:
        args.extend(["--timeout", str(manifest["timeout"])])
    if manifest.get("retries") is not None:
        args.extend(["--retries", str(manifest["retries"])])
    if manifest.get("grader_model") is not None:
        args.extend(["--grader-model", str(manifest["grader_model"])])
    if manifest.get("grader_base_url"):
        args.extend(["--grader-base-url", str(manifest["grader_base_url"])])
    if manifest.get("grader_key_file"):
        args.extend(["--grader-key-file", str(manifest["grader_key_file"])])
    if manifest.get("grader_openwebui"):
        args.append("--grader-openwebui")
    if manifest.get("source_repo"):
        args.extend(["--source-repo", str(manifest["source_repo"])])
    if manifest.get("pricing_config"):
        args.extend(["--pricing-config", str(manifest["pricing_config"])])

    budget = manifest.get("budget") or {}
    budget_flags = [
        ("max_input_tokens", "--max-input-tokens-per-case"),
        ("max_non_cached_input_tokens", "--max-non-cached-input-tokens-per-case"),
        ("max_output_tokens", "--max-output-tokens-per-case"),
        ("max_reasoning_tokens", "--max-reasoning-tokens-per-case"),
        ("max_total_tokens", "--max-total-tokens-per-case"),
        ("max_cost_usd", "--max-cost-per-case"),
        ("max_duration_seconds", "--max-duration-per-case"),
    ]
    budget_enabled = False
    for field, flag in budget_flags:
        if budget.get(field) is not None:
            args.extend([flag, str(budget[field])])
            budget_enabled = True
    if budget_enabled and budget.get("action"):
        args.extend(["--budget-action", str(budget["action"])])


def _failed_rerun_commands(manifest: dict, manifest_path: Path, workspace: Path | None) -> list[dict]:
    root = Path(manifest.get("root") or ".")
    rerun_workspace = workspace or (manifest_path.parent / "rerun-failed")
    commands: list[dict] = []
    for suite in manifest.get("suites", []):
        summary = suite.get("summary") or {}
        skill_dir = suite.get("skill_dir")
        evals_path = suite.get("evals_path")
        if not skill_dir or not evals_path:
            continue
        for failed in summary.get("failed_runs") or []:
            eval_id = failed.get("eval_id")
            agent = failed.get("agent")
            if eval_id is None or not agent:
                continue
            with_skill = bool(failed.get("with_skill", True))
            args = [
                "agent-skill-eval",
                "run",
                "--skill",
                str(root / skill_dir),
                "--evals",
                str(root / evals_path),
                "--agent",
                str(agent),
                "--workspace",
                str(rerun_workspace),
                "--eval-id",
                str(eval_id),
            ]
            _append_manifest_rerun_context(args, manifest, str(agent))
            if with_skill:
                args.append("--no-baseline")
            commands.append(
                {
                    "skill": suite.get("skill", ""),
                    "eval_id": str(eval_id),
                    "agent": str(agent),
                    "config": "with_skill" if with_skill else "without_skill",
                    "workspace": str(rerun_workspace),
                    "command": _shell_command(args),
                    "failed_assertions": failed.get("failed_assertions") or [],
                }
            )
    return commands


@app.command()
def report(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Path to workspace containing iteration dirs"),
    iteration: Optional[int] = typer.Option(None, "--iteration", "-i", help="Specific iteration to report on"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, markdown, or html"),
    show_evidence: bool = typer.Option(
        False, "--show-evidence", help="Show evidence for failed/skipped assertions from grading.json"
    ),
    failures_only: bool = typer.Option(
        False, "--failures-only", help="Only list runs that had at least one failed assertion"
    ),
):
    """Display a summary report of eval results."""
    if not workspace.exists():
        console.print(f"[red]Workspace not found: {workspace}[/red]")
        raise typer.Exit(1)
    if fmt not in ("table", "markdown", "html"):
        console.print(f"[red]Unknown format: {fmt}. Choose 'table', 'markdown', or 'html'.[/red]")
        raise typer.Exit(1)

    if iteration:
        iteration_dirs = [workspace / f"iteration-{iteration}"]
    else:
        iteration_dirs = sorted(workspace.glob("iteration-*"))

    if not iteration_dirs:
        console.print("[yellow]No iterations found[/yellow]")
        raise typer.Exit(1)

    markdown = fmt == "markdown"
    if fmt == "html":
        print(
            '<!doctype html><html><head><meta charset="utf-8">'
            "<title>agent-skill-eval report</title>"
            "<style>"
            "body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.4}"
            "table{border-collapse:collapse;margin:1rem 0;width:100%}"
            "th,td{border:1px solid #ddd;padding:.45rem;text-align:left}"
            "th{background:#f6f8fa}.iteration{margin-bottom:2rem}"
            "code{background:#f6f8fa;padding:.1rem .25rem;border-radius:4px}"
            "</style></head><body><h1>agent-skill-eval report</h1>"
        )
    for iter_dir in iteration_dirs:
        benchmark_path = iter_dir / "benchmark.json"
        if not benchmark_path.exists():
            console.print(f"[yellow]No benchmark.json in {iter_dir.name}[/yellow]")
            continue

        with open(benchmark_path, encoding="utf-8") as f:
            benchmark = json.load(f)
        summary = load_summary(iter_dir) or {}

        if fmt == "html":
            _print_html_report(iter_dir.name, benchmark, summary)
            _print_html_eval_details(iter_dir, show_evidence, failures_only=failures_only)
        elif markdown:
            _print_markdown_report(iter_dir.name, benchmark, summary)
            _print_eval_details(iter_dir, show_evidence, markdown, failures_only=failures_only)
            comparison = iter_dir / "comparison.md"
            if comparison.is_file():
                print()
                print(comparison.read_text(encoding="utf-8").strip())
        else:
            _print_table_report(iter_dir.name, benchmark, summary)
            _print_eval_details(iter_dir, show_evidence, markdown, failures_only=failures_only)
    if fmt == "html":
        print("</body></html>")


def _copy_artifact(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _archive_report_markdown(iter_dir: Path) -> str:
    benchmark_path = iter_dir / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.exists() else {}
    summary = load_summary(iter_dir) or build_summary_from_artifacts(iter_dir) or {}
    lines = [
        f"# agent-skill-eval archive: {iter_dir.name}",
        "",
        f"- skill: `{summary.get('skill_name', '')}`",
        f"- iteration: `{summary.get('iteration', '')}`",
        f"- skill_hash: `{summary.get('skill_hash', '')}`",
        f"- runs_completed: `{summary.get('runs_completed', 0)}`",
        f"- failed_runs: `{summary.get('runs_failed_grading', 0)}`",
        f"- tokens_total: `{(summary.get('tokens') or {}).get('total', 0)}`",
        "",
        "## Benchmark",
        "",
        "| " + " | ".join(REPORT_COLUMNS) + " |",
        "|" + "|".join([" --- "] * len(REPORT_COLUMNS)) + "|",
    ]
    for row in _benchmark_rows(benchmark):
        lines.append("| " + " | ".join(row) + " |")
    analysis = _analysis_lines(summary)
    if analysis:
        lines.extend(["", "## Skill analysis", ""])
        lines.extend(f"- {line}" for line in analysis)
    comparison = iter_dir / "comparison.md"
    if comparison.is_file():
        lines.extend(["", comparison.read_text(encoding="utf-8").strip()])
    return "\n".join(lines) + "\n"


@app.command(name="archive-run")
def archive_run(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Path to workspace containing iteration dirs"),
    iteration: Optional[int] = typer.Option(None, "--iteration", "-i", help="Iteration to archive (default latest)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for compact archived artifacts"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing output directory"),
    include_output: bool = typer.Option(
        False,
        "--include-output",
        help="Also copy outputs/output.txt. Disabled by default because agent output can contain secrets.",
    ),
):
    """Archive compact durable artifacts for one iteration without model calls."""
    if not workspace.exists():
        console.print(f"[red]Workspace not found: {workspace}[/red]")
        raise typer.Exit(1)
    if iteration is None:
        iteration_dirs = sorted(
            (p for p in workspace.glob("iteration-*") if p.is_dir()),
            key=lambda p: int(p.name[len("iteration-") :]) if p.name[len("iteration-") :].isdigit() else -1,
        )
        if not iteration_dirs:
            console.print(f"[red]No iterations found under {workspace}[/red]")
            raise typer.Exit(1)
        iter_dir = iteration_dirs[-1]
    else:
        iter_dir = workspace / f"iteration-{iteration}"
        if not iter_dir.exists():
            console.print(f"[red]Iteration directory not found: {iter_dir}[/red]")
            raise typer.Exit(1)

    if output.exists():
        if not force:
            console.print(f"[red]Output already exists: {output}. Pass --force to replace it.[/red]")
            raise typer.Exit(1)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copied: list[str] = []
    for name in (
        "summary.json",
        "benchmark.json",
        "cleanup.json",
        "evals_meta.json",
        "progress.jsonl",
        "comparison.json",
        "comparison.md",
    ):
        if _copy_artifact(iter_dir / name, output / name):
            copied.append(name)
    (output / "report.md").write_text(_archive_report_markdown(iter_dir), encoding="utf-8")
    copied.append("report.md")

    runs_dir = output / "runs"
    for eval_dir in sorted(iter_dir.glob("eval-*")):
        for agent_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
            for config_dir in _leaf_config_dirs(agent_dir):
                rel = config_dir.relative_to(iter_dir)
                for name in ("grading.json", "timing.json", "run_meta.json", "trajectory.json"):
                    if _copy_artifact(config_dir / name, runs_dir / rel / name):
                        copied.append(f"runs/{rel}/{name}")
                if include_output:
                    if _copy_artifact(config_dir / "outputs" / "output.txt", runs_dir / rel / "output.txt"):
                        copied.append(f"runs/{rel}/output.txt")

    manifest = {
        "source_iteration": str(iter_dir),
        "output": str(output),
        "include_output": include_output,
        "copied_files": copied,
    }
    (output / "archive-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    console.print(f"[green]Archived {len(copied)} artifact(s) to {output}[/green]")


@app.command(name="report-all")
def report_all(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Path to run-all workspace"),
    iteration: Optional[int] = typer.Option(None, "--iteration", "-i", help="Specific run-all iteration to report on"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, markdown, html, or json"),
    failures_only: bool = typer.Option(False, "--failures-only", help="Only show suites with failures or errors"),
):
    """Display a repo-level report from a run-all aggregate manifest."""
    if fmt not in ("table", "markdown", "html", "json"):
        console.print(f"[red]Unknown format: {fmt}. Choose 'table', 'markdown', 'html', or 'json'.[/red]")
        raise typer.Exit(1)

    manifest_path, manifest = _load_run_all_manifest(workspace, iteration)
    rows = _run_all_rows(manifest, failures_only=failures_only)

    if fmt == "json":
        print(json.dumps(manifest, indent=2))
        return
    if fmt == "html":
        _print_run_all_html(manifest_path, manifest, rows)
        return
    if fmt == "markdown":
        print(f"\n### {manifest_path.name}\n")
        print(_run_all_summary_line(manifest))
        print("\n| Skill | Status | Cases | Completed | Failed Runs | Errors | Duration | Tokens | Cost | Pass Rates |")
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in rows:
            print(
                f"| {row['skill']} | {row['status']} | {row['cases']} | {row['completed']} | "
                f"{row['failed']} | {row['errors']} | {row['duration']} | {row['tokens']} | "
                f"{row['cost']} | {row['pass_rates']} |"
            )
        _print_run_all_failures(rows, markdown=True)
        return

    console.print(f"[bold]{manifest_path.name}[/bold]")
    console.print(_run_all_summary_line(manifest))
    table = Table(title="Run-all Results")
    for column in [
        "Skill",
        "Status",
        "Cases",
        "Completed",
        "Failed Runs",
        "Errors",
        "Duration",
        "Tokens",
        "Cost",
        "Pass Rates",
    ]:
        table.add_column(column)
    for row in rows:
        table.add_row(
            row["skill"],
            row["status"],
            str(row["cases"]),
            str(row["completed"]),
            str(row["failed"]),
            str(row["errors"]),
            row["duration"],
            row["tokens"],
            row["cost"],
            row["pass_rates"],
        )
    console.print(table)
    _print_run_all_failures(rows, markdown=False)


@app.command()
def portfolio(
    root: Path = typer.Option(Path.cwd(), "--root", "-r", help="Directory to search for skills and eval suites"),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Optional eval workspace to include latest local run status per skill",
    ),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, markdown, or json"),
):
    """Report skill portfolio coverage, size, and latest local run status."""
    if fmt not in ("table", "markdown", "json"):
        console.print(f"[red]Unknown format: {fmt}. Choose 'table', 'markdown', or 'json'.[/red]")
        raise typer.Exit(1)

    rows = _portfolio_rows(root, workspace)
    payload = {
        "root": str(root),
        "workspace": str(workspace) if workspace else None,
        "skills": len(rows),
        "paired": sum(1 for row in rows if row["status"] == "paired"),
        "missing_evals": sum(1 for row in rows if row["status"] == "missing-evals"),
        "problems": sum(1 for row in rows if row["status"] == "problem" or row["eval_status"] == "invalid"),
        "rows": rows,
    }

    if fmt == "json":
        print(json.dumps(payload, indent=2))
        return

    if fmt == "markdown":
        print(f"\n### Skill Portfolio ({root})\n")
        print(
            f"{payload['skills']} skill(s), {payload['paired']} paired, "
            f"{payload['missing_evals']} missing evals, {payload['problems']} problem(s).\n"
        )
        print("| Skill | Status | Cases | Assertions | LLM | Side Effects | Size | Latest Run | Pass Rates | Next |")
        print("| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |")
        for row in rows:
            print(
                f"| {row['skill']} | {row['status']} | {row['eval_cases'] or 0} | "
                f"{row['assertions'] or 0} | {row['llm_assertions'] or 0} | "
                f"{_format_count_map(row['side_effect_counts'])} | {_format_skill_size(row['skill_size'])} | "
                f"{_format_portfolio_latest(row['latest_run'])} | {_format_pass_rates(row['pass_rates'])} | "
                f"{row['next']} |"
            )
        return

    console.print(f"[bold]Skill Portfolio[/bold] {root}")
    console.print(
        f"{payload['skills']} skill(s), {payload['paired']} paired, "
        f"{payload['missing_evals']} missing evals, {payload['problems']} problem(s)."
    )
    if rows:
        console.print("skills: " + ", ".join(row["skill"] for row in rows))
        console.print("next: " + ", ".join(f"{row['skill']}={row['next']}" for row in rows))
    table = Table(title="Portfolio")
    portfolio_columns = [
        "Skill",
        "Status",
        "Cases",
        "Assertions",
        "LLM",
        "Side Effects",
        "Size",
        "Latest",
        "Pass Rates",
        "Next",
    ]
    for column in portfolio_columns:
        table.add_column(column)
    for row in rows:
        table.add_row(
            row["skill"],
            row["status"],
            str(row["eval_cases"] or 0),
            str(row["assertions"] or 0),
            str(row["llm_assertions"] or 0),
            _format_count_map(row["side_effect_counts"]),
            _format_skill_size(row["skill_size"]),
            _format_portfolio_latest(row["latest_run"]),
            _format_pass_rates(row["pass_rates"]),
            row["next"],
        )
    console.print(table)


def _portfolio_rows(root: Path, workspace: Optional[Path]) -> list[dict]:
    suites = _discover_eval_suites(root)
    skills = _discover_skills(root)
    rows = []
    pairs = sorted(
        _skill_eval_pairs(root, suites, skills),
        key=lambda item: (item.get("skill") or "", item["skill_dir"]),
    )
    for pair in pairs:
        skill_dir = root / pair["skill_dir"]
        skill_size = SkillInstaller(skill_dir).size_metrics() if (skill_dir / "SKILL.md").exists() else None
        row = {
            **pair,
            "eval_status": "missing" if not pair.get("evals_path") else "unknown",
            "eval_cases": None,
            "assertions": None,
            "deterministic_assertions": None,
            "llm_assertions": None,
            "side_effect_counts": {},
            "strict_warnings": [],
            "validation_error": None,
            "skill_size": skill_size,
            "latest_run": _latest_portfolio_run(workspace, pair.get("skill")) if workspace else None,
            "pass_rates": {},
        }
        if pair.get("evals_path"):
            try:
                _, report = load_and_validate_suite(root / pair["evals_path"])
                row.update(
                    {
                        "eval_status": "valid",
                        "eval_cases": report.selected_eval_count,
                        "assertions": report.total_assertions,
                        "deterministic_assertions": len(report.deterministic_assertions),
                        "llm_assertions": len(report.llm_assertions),
                        "side_effect_counts": report.side_effect_counts,
                        "strict_warnings": list(report.strict_warnings),
                    }
                )
            except EvalSuiteValidationError as e:
                row["eval_status"] = "invalid"
                row["validation_error"] = str(e)
        if row["latest_run"]:
            row["pass_rates"] = row["latest_run"].get("pass_rates") or {}
        row["next"] = _portfolio_next(row)
        rows.append(row)
    return rows


def _latest_portfolio_run(workspace: Optional[Path], skill: str | None) -> dict | None:
    if not workspace or not skill:
        return None
    skill_workspace = workspace / f"{skill}-workspace"
    if not skill_workspace.exists():
        return None
    iteration_dirs = sorted(
        (path for path in skill_workspace.glob("iteration-*") if path.is_dir()),
        key=lambda path: _iteration_suffix(path),
    )
    if not iteration_dirs:
        return None
    latest = iteration_dirs[-1]
    summary = load_summary(latest) or build_summary_from_artifacts(latest) or {}
    return {
        "iteration": _iteration_suffix(latest),
        "path": str(latest),
        "runs_completed": summary.get("runs_completed", 0),
        "runs_failed_grading": summary.get("runs_failed_grading", 0),
        "errors": len(summary.get("errors") or []),
        "flaky_count": ((summary.get("stability") or {}).get("flaky_count") or 0),
        "pass_rates": summary.get("pass_rates") or {},
        "tokens": summary.get("tokens") or {},
        "cost": summary.get("cost") or {},
    }


def _iteration_suffix(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _portfolio_next(row: dict) -> str:
    if row["status"] == "missing-evals":
        return "Add evals"
    if row["status"] == "problem":
        return "Fix skill/eval pairing"
    if row["eval_status"] == "invalid":
        return "Fix eval validation"
    if not row.get("eval_cases"):
        return "Add eval cases"
    if row.get("strict_warnings"):
        return "Tighten assertions"
    latest = row.get("latest_run")
    if not latest:
        return "Run evals"
    if latest.get("errors") or latest.get("runs_failed_grading"):
        return "Debug failures"
    if latest.get("flaky_count"):
        return "Investigate flakes"
    if (row.get("llm_assertions") or 0) > (row.get("deterministic_assertions") or 0):
        return "Add deterministic checks"
    return "Keep watching"


def _format_count_map(counts: dict | None) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _format_skill_size(size: dict | None) -> str:
    if not size:
        return "-"
    return f"{size.get('words', 0)}w/{size.get('files', 0)}f"


def _format_portfolio_latest(latest: dict | None) -> str:
    if not latest:
        return "-"
    parts = [f"it-{latest.get('iteration')}", f"{latest.get('runs_completed', 0)} runs"]
    if latest.get("runs_failed_grading"):
        parts.append(f"{latest['runs_failed_grading']} failed")
    if latest.get("errors"):
        parts.append(f"{latest['errors']} errors")
    if latest.get("flaky_count"):
        parts.append(f"{latest['flaky_count']} flaky")
    return ", ".join(parts)


@app.command(name="rerun-failed")
def rerun_failed(
    manifest_path: Path = typer.Option(..., "--from", "-f", help="Path to run-all-iteration-N.json"),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace base to use in printed rerun commands (default: <manifest-dir>/rerun-failed)",
    ),
    output_format: str = typer.Option("shell", "--format", help="Output format: shell, table, or json"),
):
    """Print targeted commands for failed eval cases from a run-all manifest."""
    if output_format not in ("shell", "table", "json"):
        console.print(f"[red]Unknown format: {output_format}. Choose 'shell', 'table', or 'json'.[/red]")
        raise typer.Exit(1)
    manifest = _load_run_all_manifest_file(manifest_path)
    commands = _failed_rerun_commands(manifest, manifest_path, workspace)

    if output_format == "json":
        print(json.dumps({"manifest": str(manifest_path), "commands": commands}, indent=2))
        return
    if not commands:
        console.print("[green]No failed eval runs found in manifest[/green]")
        return
    if output_format == "shell":
        for item in commands:
            print(item["command"])
        return

    table = Table(title=f"Failed eval reruns from {manifest_path.name}")
    table.add_column("Skill", style="cyan")
    table.add_column("Eval", style="green")
    table.add_column("Agent", style="magenta")
    table.add_column("Config", style="yellow")
    table.add_column("Command")
    for item in commands:
        table.add_row(item["skill"], item["eval_id"], item["agent"], item["config"], item["command"])
    console.print(table)


@app.command()
def compare(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Path to workspace containing iteration dirs"),
    iteration_a: int = typer.Argument(..., help="Baseline iteration number"),
    iteration_b: int = typer.Argument(..., help="Comparison iteration number"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, markdown, or json"),
):
    """Compare benchmark results between two iterations of the same skill."""
    if fmt not in ("table", "markdown", "json"):
        console.print(f"[red]Unknown format: {fmt}. Choose 'table', 'markdown', or 'json'.[/red]")
        raise typer.Exit(1)
    benchmarks = {}
    summaries = {}
    for n in (iteration_a, iteration_b):
        iter_dir = workspace / f"iteration-{n}"
        path = iter_dir / "benchmark.json"
        if not path.exists():
            console.print(f"[red]No benchmark.json for iteration-{n} in {workspace}[/red]")
            raise typer.Exit(1)
        with open(path, encoding="utf-8") as f:
            benchmarks[n] = json.load(f)
        summaries[n] = load_summary(iter_dir) or build_summary_from_artifacts(iter_dir) or {}

    hash_a = summaries[iteration_a].get("skill_hash")
    hash_b = summaries[iteration_b].get("skill_hash")
    if fmt != "json" and hash_a and hash_b and hash_a != hash_b:
        console.print(
            f"[yellow]Warning: skill hashes differ between iterations "
            f"({hash_a[:12]} vs {hash_b[:12]}). Deltas include skill content changes.[/yellow]"
        )

    rows = _compare_rows(benchmarks[iteration_a], benchmarks[iteration_b])
    failed_a = _failed_assertion_counter(summaries[iteration_a])
    failed_b = _failed_assertion_counter(summaries[iteration_b])
    changed_files = _changed_skill_files(summaries[iteration_a], summaries[iteration_b])
    size_a = summaries[iteration_a].get("skill_size") or {}
    size_b = summaries[iteration_b].get("skill_size") or {}
    bytes_a = size_a.get("bytes")
    bytes_b = size_b.get("bytes")
    words_a = size_a.get("words")
    words_b = size_b.get("words")
    skill_md_words_a = (size_a.get("skill_md") or {}).get("words")
    skill_md_words_b = (size_b.get("skill_md") or {}).get("words")

    payload = {
        "workspace": str(workspace),
        "iteration_a": iteration_a,
        "iteration_b": iteration_b,
        "skill_hash_a": hash_a,
        "skill_hash_b": hash_b,
        "skill_size_a": size_a,
        "skill_size_b": size_b,
        "skill_size_delta": {
            "bytes": (bytes_b - bytes_a) if bytes_a is not None and bytes_b is not None else None,
            "words": (words_b - words_a) if words_a is not None and words_b is not None else None,
            "skill_md_words": (
                (skill_md_words_b - skill_md_words_a)
                if skill_md_words_a is not None and skill_md_words_b is not None
                else None
            ),
        },
        "changed_skill_files": changed_files,
        "failed_assertion_delta": {
            assertion: failed_b.get(assertion, 0) - failed_a.get(assertion, 0)
            for assertion in sorted(set(failed_a) | set(failed_b))
        },
        "rows": rows,
    }

    if fmt == "json":
        print(json.dumps(payload, indent=2))
        return

    def pct(value) -> str:
        return f"{value:.1%}" if value is not None else "-"

    def pct_delta(value) -> str:
        return f"{value:+.1%}" if value is not None else "-"

    def num(value, suffix: str = "") -> str:
        return f"{value:+.1f}{suffix}" if value is not None else "-"

    if fmt == "markdown":
        print(f"\n### iteration-{iteration_a} vs iteration-{iteration_b}\n")
        print(
            "| Configuration | Pass A | Pass B | Pass Δ | Full Pass A | Full Pass B | "
            "pass@k A | pass@k B | Time Δ (s) | Tokens Δ | Cost Δ |"
        )
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            print(
                f"| {row['configuration']} | {pct(row['pass_rate_a'])} | {pct(row['pass_rate_b'])} | "
                f"{pct_delta(row['pass_rate_delta'])} | "
                f"{pct(row['full_pass_a'])} | {pct(row['full_pass_b'])} | "
                f"{pct(row['pass_at_k_a'])} | {pct(row['pass_at_k_b'])} | "
                f"{num(row['time_delta'])} | {num(row['tokens_delta'])} | {row['cost_delta']} |"
            )
        print()
        print(f"- skill_hash: `{str(hash_a or '')[:12]}` -> `{str(hash_b or '')[:12]}`")
        skill_size_words_line = (
            f"- skill_size words: {size_a.get('words', '-')} -> {size_b.get('words', '-')} "
            f"({payload['skill_size_delta']['words']:+} words)"
            if payload["skill_size_delta"]["words"] is not None
            else "- skill_size words: n/a"
        )
        print(skill_size_words_line)
        print(
            f"- SKILL.md words: {(size_a.get('skill_md') or {}).get('words', '-')} -> "
            f"{(size_b.get('skill_md') or {}).get('words', '-')} "
            f"({payload['skill_size_delta']['skill_md_words']:+} words)"
            if payload["skill_size_delta"]["skill_md_words"] is not None
            else "- SKILL.md words: n/a"
        )
        if changed_files:
            print("- changed skill files: " + ", ".join(f"`{path}`" for path in changed_files))
        if payload["failed_assertion_delta"]:
            print("- failed assertion deltas:")
            for assertion, delta in payload["failed_assertion_delta"].items():
                print(f"  - {delta:+}: {assertion}")
        return

    table = Table(title=f"iteration-{iteration_a} vs iteration-{iteration_b}")
    for column, style, min_width in [
        ("Config", "cyan", None),
        (f"Pass {iteration_a}", "green", 6),
        (f"Pass {iteration_b}", "green", 6),
        ("Pass Δ", "magenta", 6),
        ("Full Δ", "magenta", 6),
        ("Time Δ", "yellow", None),
        ("Tokens Δ", "blue", None),
        ("Cost Δ", "white", None),
    ]:
        table.add_column(column, style=style, no_wrap=True, min_width=min_width)
    for row in rows:
        full_delta = (
            row["full_pass_b"] - row["full_pass_a"]
            if row["full_pass_a"] is not None and row["full_pass_b"] is not None
            else None
        )
        table.add_row(
            row["configuration"],
            pct(row["pass_rate_a"]),
            pct(row["pass_rate_b"]),
            pct_delta(row["pass_rate_delta"]),
            pct_delta(full_delta),
            num(row["time_delta"]),
            num(row["tokens_delta"]),
            row["cost_delta"],
        )
    console.print(table)
    if rows:
        deltas = []
        for row in rows:
            delta = pct_delta(row["pass_rate_delta"])
            deltas.append(f"{row['configuration']} {delta}")
        console.print("pass_delta: " + ", ".join(deltas))
    if size_a or size_b:
        skill_md_a = (size_a.get("skill_md") or {}).get("words", "-")
        skill_md_b = (size_b.get("skill_md") or {}).get("words", "-")
        console.print(
            f"skill_size: words {size_a.get('words', '-')} -> {size_b.get('words', '-')} "
            f"(SKILL.md {skill_md_a} -> {skill_md_b})"
        )
    if changed_files:
        console.print("changed skill files: " + ", ".join(changed_files))
    if payload["failed_assertion_delta"]:
        console.print("failed assertion deltas:")
        for assertion, delta in payload["failed_assertion_delta"].items():
            console.print(f"  {delta:+}: {assertion}")


@app.command()
def status(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Path to workspace containing iteration dirs"),
    iteration: Optional[int] = typer.Option(None, "--iteration", "-i", help="Iteration to show (default: latest)"),
):
    """Show the latest run's summary (pass rate, failures, tokens, cost, cleanup) without any model call.

    Reads the iteration's ``summary.json``; for runs that predate it, a
    reduced summary is rebuilt from the saved artifacts.
    """
    if not workspace.exists():
        console.print(f"[red]Workspace not found: {workspace}[/red]")
        raise typer.Exit(1)

    if iteration is not None:
        iter_dir = workspace / f"iteration-{iteration}"
        if not iter_dir.exists():
            console.print(f"[red]Iteration directory not found: {iter_dir}[/red]")
            raise typer.Exit(1)
    else:
        iteration_dirs = sorted(
            (p for p in workspace.glob("iteration-*") if p.is_dir()),
            key=lambda p: int(p.name[len("iteration-") :]) if p.name[len("iteration-") :].isdigit() else -1,
        )
        if not iteration_dirs:
            console.print(f"[yellow]No iterations found under {workspace}[/yellow]")
            raise typer.Exit(1)
        iter_dir = iteration_dirs[-1]

    summary = load_summary(iter_dir)
    if summary is None:
        summary = build_summary_from_artifacts(iter_dir)
        if summary is None:
            console.print(f"[red]No summary.json and no readable artifacts in {iter_dir}[/red]")
            raise typer.Exit(1)
        console.print("[yellow]No summary.json (run predates it); rebuilt a reduced summary from artifacts.[/yellow]")

    console.print(f"\n[bold]{iter_dir}[/bold]")
    # `or`-defaults everywhere below: summary.json fields can be
    # present-but-null (hand-edited or partially written files).
    console.print(
        f"skill={summary.get('skill_name')} iteration={summary.get('iteration')} "
        f"agents={','.join(summary.get('agents') or [])} "
        f"eval_ids={','.join(str(e) for e in summary.get('eval_ids') or [])}"
    )
    if summary.get("skill_hash"):
        console.print(f"skill_hash={summary['skill_hash']}")
    if summary.get("side_effect_levels"):
        console.print(f"side_effect_levels={','.join(summary.get('side_effect_levels') or [])}")
    console.print(
        f"runs: {summary.get('runs_completed') or 0} completed, "
        f"{summary.get('runs_failed_grading') or 0} with failed assertions, "
        f"{len(summary.get('errors') or [])} errored, "
        f"{len(summary.get('skipped_runs') or [])} skipped; "
        f"duration {summary.get('duration_seconds') or 0:.0f}s"
    )

    pass_rates = summary.get("pass_rates") or {}
    if pass_rates:
        table = Table(title="Pass rates")
        table.add_column("Configuration", style="cyan")
        table.add_column("Pass rate", style="green")
        for config, rate in pass_rates.items():
            table.add_row(config, f"{rate:.1%}" if rate is not None else "-")
        console.print(table)

    tokens = summary.get("tokens") or {}
    if tokens:
        reasoning = tokens.get("reasoning_output")
        reasoning_str = f" (reasoning={reasoning:,})" if reasoning is not None else ""
        console.print(
            f"tokens: input={tokens.get('input', 0):,} "
            f"(cached={tokens.get('cached', 0):,}, {tokens.get('cached_pct', 0):.0%}; "
            f"non-cached={tokens.get('non_cached_input', 0):,}) "
            f"output={tokens.get('output', 0):,}{reasoning_str} total={tokens.get('total', 0):,}"
        )
    cost = summary.get("cost") or {}
    total_cost = cost.get("total_usd")
    cost_str = f"{total_cost:.4f} USD" if total_cost is not None else "unavailable"
    console.print(f"cost: {cost_str} ({cost.get('runs_with_cost', 0)}/{cost.get('runs_total', 0)} runs reported cost)")

    for failed in summary.get("failed_runs") or []:
        config = "with_skill" if failed.get("with_skill") else "without_skill"
        console.print(
            f"  [red]FAILED {failed.get('eval_id')}/{failed.get('agent')}/{config}: "
            f"{', '.join(failed.get('failed_assertions') or []) or 'see grading.json'}[/red]"
        )
    for skipped in summary.get("skipped_runs") or []:
        console.print(f"  [yellow]SKIPPED {skipped.get('run')}: {skipped.get('reason')}[/yellow]")
    for error in summary.get("errors") or []:
        console.print(f"  [red]ERROR {error.get('run')}: {error.get('error')}[/red]")

    budget = summary.get("budget") or {}
    for exceeded in budget.get("exceeded_runs") or []:
        config = "with_skill" if exceeded.get("with_skill") else "without_skill"
        console.print(
            f"  [yellow]BUDGET {exceeded.get('eval_id')}/{exceeded.get('agent')}/{config}: "
            f"{exceeded.get('reason')}[/yellow]"
        )

    stability = summary.get("stability") or {}
    flaky = stability.get("flaky_groups") or []
    if flaky:
        console.print(f"[yellow]stability: {len(flaky)} flaky group(s)[/yellow]")
        for group in flaky:
            config = "with_skill" if group.get("with_skill") else "without_skill"
            console.print(
                f"  [yellow]{group.get('eval_id')}/{group.get('agent')}/{config}: "
                f"{group.get('full_passes')}/{group.get('runs')} full passes[/yellow]"
            )
    elif stability.get("groups"):
        console.print(f"stability: no flaky groups across {len(stability.get('groups') or [])} group(s)")

    cleanup_info = summary.get("cleanup") or {}
    branches = cleanup_info.get("remote_branches") or []
    prs = cleanup_info.get("pr_numbers") or []
    mrs = cleanup_info.get("merge_request_numbers") or []
    if branches or prs or mrs:
        console.print(
            f"[yellow]side effects recorded: {len(branches)} remote branch(es), "
            f"{len(prs)} PR(s), {len(mrs)} MR(s) — "
            f"run 'agent-skill-eval cleanup' to remove them[/yellow]"
        )
    else:
        console.print("side effects recorded: none")

    hooks_info = summary.get("hooks") or {}
    for stage, records in hooks_info.items():
        failures = [r for r in records if r.get("exit_code") != 0]
        state = f"[red]{len(failures)} failed[/red]" if failures else "ok"
        console.print(f"hooks ({stage}): {len(records)} ran, {state}")


@app.command()
def validate(
    evals: Path = typer.Argument(..., help="Path to an evals.json file to validate"),
    eval_ids: list[str] = typer.Option(
        [],
        "--eval-id",
        help="Only validate the eval case with this id. Repeatable. "
        "Pairs with 'run --eval-id' for a cheap targeted inner loop.",
    ),
    side_effect_levels: list[str] = typer.Option(
        [],
        "--side-effect-level",
        help=(
            "Only validate eval cases with this side-effect level. "
            f"Repeatable. Choices: {', '.join(SIDE_EFFECT_LEVELS)}"
        ),
    ),
    strict: bool = typer.Option(False, "--strict", help="Fail on low-signal eval suite design warnings"),
):
    """Validate an evals.json file against the eval suite schema."""
    try:
        _, report = load_and_validate_suite(evals, eval_ids=eval_ids, side_effect_levels=side_effect_levels)
    except EvalSuiteValidationError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    _print_validation_report(report)
    if strict and report.strict_warnings:
        raise typer.Exit(1)


@app.command(name="validate-all")
def validate_all(
    root: Path = typer.Option(Path.cwd(), "--root", "-r", help="Directory to recursively search for eval suites"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format: table or json"),
    side_effect_levels: list[str] = typer.Option(
        [],
        "--side-effect-level",
        help=(
            "Only validate eval cases with this side-effect level. "
            f"Repeatable. Choices: {', '.join(SIDE_EFFECT_LEVELS)}"
        ),
    ),
    strict: bool = typer.Option(False, "--strict", help="Fail on low-signal eval suite design warnings"),
):
    """Validate every discovered evals.json under a directory."""
    _ensure_output_format(output_format)
    suites = _discover_eval_suites(root)
    if not suites:
        if output_format == "json":
            typer.echo(json.dumps({"root": str(root), "suites": [], "failures": 1}, indent=2))
        else:
            console.print(f"[red]No evals.json files found under {root}[/red]")
        raise typer.Exit(1)

    failures = 0
    failure_details: list[tuple[str, str]] = []
    valid_skills: list[str] = []
    results: list[dict] = []
    table = Table(title=f"Eval suite validation under {root}")
    table.add_column("Status", style="bold")
    table.add_column("Skill", style="cyan")
    table.add_column("Evals file", style="green")
    table.add_column("Cases", style="yellow", justify="right")
    table.add_column("Assertions", style="yellow", justify="right")
    table.add_column("LLM", style="yellow", justify="right")
    table.add_column("Side effects / Error")

    for path in suites:
        rel = str(path.relative_to(root))
        try:
            _, report = load_and_validate_suite(path, side_effect_levels=side_effect_levels)
        except EvalSuiteValidationError as e:
            failures += 1
            message = str(e).replace("\n", "; ")
            failure_details.append((rel, message))
            results.append({"status": "fail", "path": rel, "error": message})
            table.add_row("[red]fail[/red]", "-", rel, "-", "-", "-", message)
            continue

        pair = _skill_eval_pair(root, path, skill_name=report.skill_name)
        if pair["problems"]:
            failures += 1
            message = "; ".join(pair["problems"])
            failure_details.append((rel, message))
            results.append({"status": "fail", "path": rel, "skill": report.skill_name, "error": message, "pair": pair})
            table.add_row("[red]fail[/red]", report.skill_name, rel, "-", "-", "-", message)
            continue

        strict_warnings = list(report.strict_warnings)
        strict_failed = strict and bool(strict_warnings)
        if strict_failed:
            failures += 1
            failure_details.append((rel, "; ".join(strict_warnings)))
        else:
            valid_skills.append(report.skill_name)
        results.append(
            {
                "status": "strict_fail" if strict_failed else "ok",
                "skill": report.skill_name,
                "path": rel,
                "cases": report.selected_eval_count,
                "assertions": report.total_assertions,
                "llm_assertions": len(report.llm_assertions),
                "side_effect_counts": report.side_effect_counts,
                "strict_warnings": strict_warnings,
                "pair": pair,
            }
        )
        side_effects = ", ".join(
            f"{level}={count}" for level, count in sorted(report.side_effect_counts.items()) if count
        )
        detail = side_effects or "-"
        if strict_warnings:
            detail = f"{detail}; strict: {'; '.join(strict_warnings)}"
        table.add_row(
            "[yellow]strict[/yellow]" if strict_failed else "[green]ok[/green]",
            report.skill_name,
            rel,
            str(report.selected_eval_count),
            str(report.total_assertions),
            str(len(report.llm_assertions)),
            detail,
        )

    if output_format == "json":
        typer.echo(json.dumps({"root": str(root), "suites": results, "failures": failures}, indent=2))
        if failures:
            raise typer.Exit(1)
        return

    console.print(table)
    if failures:
        for rel, message in failure_details:
            console.print(f"[red]{rel}: {message}[/red]")
        console.print(f"[red]{failures}/{len(suites)} eval suite(s) failed validation[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Validated skills: {', '.join(valid_skills)}[/green]")
    console.print(f"[green]All {len(suites)} eval suite(s) valid[/green]")


def _print_validation_report(report: EvalSuiteValidationReport) -> None:
    selected = " (selected by --eval-id)" if report.selected_by_eval_id else ""
    console.print(
        f"[green]Valid: {report.selected_eval_count} eval(s) for skill '{report.skill_name}'{selected}[/green]"
    )
    if report.side_effect_counts:
        side_effects = ", ".join(f"{level}={count}" for level, count in sorted(report.side_effect_counts.items()))
        console.print(f"[cyan]Side-effect levels: {side_effects}[/cyan]")
    llm_assertions = report.llm_assertions
    if llm_assertions:
        console.print(
            f"[yellow]{len(llm_assertions)}/{report.total_assertions} assertion(s) match no deterministic "
            f"pattern and will be graded by the LLM rubric (needs an API key; verdicts can vary):[/yellow]"
        )
        for item in llm_assertions:
            console.print(f"  [dim]{item.eval_id}: {item.assertion}[/dim]")
    elif report.total_assertions:
        console.print(f"[green]All {report.total_assertions} assertion(s) match deterministic patterns[/green]")
    if report.strict_warnings:
        console.print("[yellow]Strict eval design warnings:[/yellow]")
        for warning in report.strict_warnings:
            console.print(f"  [yellow]- {warning}[/yellow]")


@app.command(name="list")
def list_suites(
    root: Path = typer.Option(Path.cwd(), "--root", "-r", help="Directory to search for eval suites and skills"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format: table or json"),
):
    """List discoverable skills (SKILL.md) and eval suites (evals.json)."""
    _ensure_output_format(output_format)
    suites = _discover_eval_suites(root)
    skills = _discover_skills(root)
    suite_rows = _eval_suite_listing(root, suites)
    skill_rows = [str(path.parent.relative_to(root)) for path in skills]
    pair_rows = _skill_eval_pairs(root, suites, skills)

    if output_format == "json":
        typer.echo(
            json.dumps(
                {"root": str(root), "eval_suites": suite_rows, "skills": skill_rows, "pairs": pair_rows},
                indent=2,
            )
        )
        return

    seen: set[Path] = set()
    table = Table(title=f"Eval suites under {root}")
    table.add_column("Skill", style="cyan")
    table.add_column("Evals file", style="green")
    table.add_column("Cases", style="yellow")
    for path in suites:
        if path in seen:
            continue
        seen.add(path)
        row = next(item for item in suite_rows if item["path"] == str(path.relative_to(root)))
        table.add_row(row["skill"], row["path"], str(row["cases"]) if row["cases"] is not None else "-")
    if seen:
        console.print(table)
    else:
        console.print(f"[yellow]No evals.json files found under {root}[/yellow]")

    seen_skills: set[str] = set()
    skill_table = Table(title=f"Skills under {root}")
    skill_table.add_column("Skill dir", style="cyan")
    for path in skill_rows:
        if path in seen_skills:
            continue
        seen_skills.add(path)
        skill_table.add_row(path)
    if seen_skills:
        console.print(skill_table)

    pair_table = Table(title=f"Skill/eval pairs under {root}")
    pair_table.add_column("Status", style="bold")
    pair_table.add_column("Skill", style="cyan")
    pair_table.add_column("Skill dir", style="green")
    pair_table.add_column("Evals file", style="green")
    pair_table.add_column("Problems")
    for pair in pair_rows:
        if pair["status"] == "paired":
            status = "[green]paired[/green]"
        elif pair["status"] == "missing-evals":
            status = "[yellow]missing evals[/yellow]"
        elif pair["status"] == "external-suite":
            status = "[cyan]external suite[/cyan]"
        else:
            status = "[red]problem[/red]"
        pair_table.add_row(
            status,
            pair.get("skill") or "-",
            pair.get("skill_dir") or "-",
            pair.get("evals_path") or "-",
            "; ".join(pair.get("problems") or []) or "-",
        )
    if pair_rows:
        console.print(pair_table)


def _ensure_output_format(output_format: str) -> None:
    if output_format not in {"table", "json"}:
        console.print(f"[red]Unknown output format: {output_format}. Choose from: table, json[/red]")
        raise typer.Exit(1)


def _eval_suite_listing(root: Path, suites: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[Path] = set()
    for path in suites:
        if path in seen:
            continue
        seen.add(path)
        rel = str(path.relative_to(root))
        try:
            data = json.loads(path.read_text())
            rows.append(
                {
                    "skill": data.get("skill_name", "?"),
                    "path": rel,
                    "cases": len(data.get("evals", [])),
                    "error": None,
                }
            )
        except (json.JSONDecodeError, OSError) as e:
            rows.append({"skill": "(unparseable)", "path": rel, "cases": None, "error": str(e)})
    return rows


def _skill_eval_pairs(root: Path, suites: list[Path], skills: list[Path]) -> list[dict]:
    rows = [_skill_eval_pair(root, suite) for suite in suites]
    suite_dirs = {suite.parent.parent.resolve() for suite in suites}
    for skill in skills:
        skill_dir = skill.parent.resolve()
        if skill_dir not in suite_dirs:
            rel = str(skill.parent.relative_to(root))
            rows.append(
                {
                    "status": "missing-evals",
                    "skill": skill.parent.name,
                    "skill_dir": rel,
                    "evals_path": None,
                    "problems": ["No evals/evals.json found next to this skill"],
                }
            )
    return rows


def _skill_eval_pair(root: Path, suite: Path, skill_name: str | None = None) -> dict:
    skill_dir = suite.parent.parent
    skill_md = skill_dir / "SKILL.md"
    rel_skill_dir = str(skill_dir.relative_to(root))
    rel_suite = str(suite.relative_to(root))
    problems: list[str] = []

    if skill_name is None:
        try:
            data = json.loads(suite.read_text())
            raw_skill_name = data.get("skill_name")
            skill_name = str(raw_skill_name) if raw_skill_name else None
        except (json.JSONDecodeError, OSError):
            skill_name = None

    plugin_style = skill_dir.parent.name == "skills"
    if not skill_md.exists():
        if plugin_style:
            problems.append("Missing SKILL.md next to plugin-style eval suite")
        return {
            "status": "problem" if problems else "external-suite",
            "skill": skill_name,
            "skill_dir": rel_skill_dir,
            "evals_path": rel_suite,
            "problems": problems,
        }

    problems.extend(SkillInstaller(skill_dir).frontmatter_problems())
    if skill_name and skill_name != skill_dir.name:
        problems.append(f"eval suite skill_name {skill_name!r} does not match skill directory {skill_dir.name!r}")

    return {
        "status": "problem" if problems else "paired",
        "skill": skill_name or skill_dir.name,
        "skill_dir": rel_skill_dir,
        "evals_path": rel_suite,
        "problems": problems,
    }


def _discover_eval_suites(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("evals.json") if path.parent.name == "evals" and not _is_discovery_ignored(path)
    )


def _discover_skills(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("SKILL.md") if not _is_discovery_ignored(path))


def _is_discovery_ignored(path: Path) -> bool:
    return any(part in DISCOVERY_IGNORED_DIRS for part in path.parts)


SKILL_MD_TEMPLATE = """\
---
name: {skill_name}
description: One-line description used by the agent to decide when to trigger this skill.
license: MIT
compatibility: opencode, claude-code, codex
---

## What I do

Describe the skill's behavior here.

## When to use me

Trigger on phrases like:
- "..."

## Steps

1. ...
"""


INIT_TEMPLATES = ("basic", "review-diff", "commit-pr", "config-validator")


def _init_template_payload(skill_name: str, template: str) -> tuple[dict, dict[str, str]]:
    if template == "basic":
        return (
            {
                "skill_name": skill_name,
                "evals": [
                    {
                        "id": "explicit-invoke",
                        "prompt": f"Use the ${skill_name} skill to do its primary function",
                        "expected_output": "Description of what success looks like",
                        "force_skill_invocation": True,
                        "side_effect_level": "local-only",
                        "assertions": [
                            {
                                "type": "file_contains",
                                "contains": "expected content",
                                "text": "The output contains expected content",
                            }
                        ],
                    },
                    {
                        "id": "natural-trigger",
                        "prompt": "Do the workflow this skill is meant to handle, without naming the skill explicitly.",
                        "expected_output": "The skill should trigger naturally.",
                        "side_effect_level": "local-only",
                        "assertions": [
                            {
                                "type": "file_contains",
                                "contains": "expected content",
                                "text": "The output contains expected content",
                            }
                        ],
                    },
                    {
                        "id": "negative-control",
                        "prompt": "Show me the git log for the last 10 commits",
                        "expected_output": "The skill should NOT trigger.",
                        "should_trigger": False,
                        "side_effect_level": "static",
                        "assertions": [
                            "A new git branch was created",
                            "A git commit was created",
                        ],
                    },
                ],
            },
            {},
        )

    if template == "review-diff":
        return (
            {
                "skill_name": skill_name,
                "evals": [
                    {
                        "id": "explicit-review",
                        "prompt": f"Use the ${skill_name} skill to review changes.diff. Do not modify files.",
                        "expected_output": "A concise review with file/line findings and no workspace changes.",
                        "files": ["files/changes.diff"],
                        "force_skill_invocation": True,
                        "side_effect_level": "local-only",
                        "assertions": [
                            {
                                "type": "file_contains",
                                "contains": "Findings",
                                "text": "The output includes a findings section",
                            },
                            {
                                "type": "file_not_exists",
                                "path": "unexpected-change.txt",
                                "text": "The review does not create unrelated files",
                            },
                        ],
                    },
                    {
                        "id": "negative-control",
                        "prompt": "Summarize what files are present. Do not review code.",
                        "expected_output": "The skill should NOT trigger.",
                        "files": ["files/changes.diff"],
                        "should_trigger": False,
                        "side_effect_level": "static",
                        "assertions": [
                            "A new git branch was created",
                            "A git commit was created",
                        ],
                    },
                ],
            },
            {
                "files/changes.diff": """diff --git a/app.py b/app.py
index 1111111..2222222 100644
--- a/app.py
+++ b/app.py
@@ -1,3 +1,6 @@
 def handler(request):
+    if request.args.get("debug"):
+        print(request)
     return "ok"
""",
            },
        )

    if template == "commit-pr":
        return (
            {
                "skill_name": skill_name,
                "evals": [
                    {
                        "id": "explicit-commit-pr",
                        "prompt": f"Use the ${skill_name} skill to commit the staged change, push it, and open a PR.",
                        "expected_output": "The agent creates a branch, commit, push, and pull request.",
                        "files": ["files/sample_change.py"],
                        "force_skill_invocation": True,
                        "stage_files": True,
                        "side_effect_level": "live-external",
                        "assertions": [
                            "A new git branch was created",
                            "A git commit was created",
                            "Changes were pushed to remote",
                            "A pull request was created",
                        ],
                    },
                    {
                        "id": "natural-trigger",
                        "prompt": "Please get this change into a PR.",
                        "expected_output": "The skill should trigger naturally and create a PR.",
                        "files": ["files/sample_change.py"],
                        "stage_files": True,
                        "side_effect_level": "live-external",
                        "assertions": [
                            "A new git branch was created",
                            "A git commit was created",
                            "Changes were pushed to remote",
                            "A pull request was created",
                        ],
                    },
                    {
                        "id": "status-only-negative-control",
                        "prompt": "Show me git status and do not change anything.",
                        "expected_output": "The skill should NOT trigger.",
                        "should_trigger": False,
                        "side_effect_level": "static",
                        "assertions": [
                            "A new git branch was created",
                            "A git commit was created",
                            "Changes were pushed to remote",
                            "A pull request was created",
                        ],
                    },
                ],
            },
            {
                "files/sample_change.py": """def changed_behavior():
    return "replace this fixture with a realistic change"
""",
            },
        )

    if template == "config-validator":
        return (
            {
                "skill_name": skill_name,
                "evals": [
                    {
                        "id": "explicit-valid-config",
                        "prompt": f"Use the ${skill_name} skill to validate app_config.json.",
                        "expected_output": "The agent validates the config and reports success.",
                        "files": ["files/app_config.json"],
                        "force_skill_invocation": True,
                        "side_effect_level": "local-only",
                        "assertions": [
                            "The output contains `valid`",
                            "The output is valid JSON",
                        ],
                    },
                    {
                        "id": "negative-control",
                        "prompt": "Tell me the filename of the config only.",
                        "expected_output": "The skill should NOT trigger.",
                        "files": ["files/app_config.json"],
                        "should_trigger": False,
                        "side_effect_level": "static",
                        "assertions": [
                            "A new git branch was created",
                            "A git commit was created",
                        ],
                    },
                ],
            },
            {
                "files/app_config.json": json.dumps({"name": "example", "enabled": True}, indent=2) + "\n",
            },
        )

    raise ValueError(f"Unknown template: {template}. Choose from: {', '.join(INIT_TEMPLATES)}")


@app.command()
def init(
    skill_name: str = typer.Argument(..., help="Name of the skill to create eval structure for"),
    output: Path = typer.Option(Path.cwd(), "--output", "-o", help="Output directory"),
    layout: str = typer.Option("standalone", "--layout", help="Scaffold layout: standalone or plugin"),
    plugin: Optional[str] = typer.Option(None, "--plugin", help="Plugin name for --layout plugin"),
    template: str = typer.Option(
        "basic",
        "--template",
        "-t",
        help=f"Eval scaffold template: {', '.join(INIT_TEMPLATES)}",
    ),
):
    """Initialize an eval structure (and a SKILL.md template) for a skill."""
    if template not in INIT_TEMPLATES:
        console.print(f"[red]Unknown template: {template}. Choose from: {', '.join(INIT_TEMPLATES)}[/red]")
        raise typer.Exit(1)
    if layout not in {"standalone", "plugin"}:
        console.print(f"[red]Unknown layout: {layout}. Choose from: standalone, plugin[/red]")
        raise typer.Exit(1)
    if layout == "plugin" and not plugin:
        console.print("[red]--plugin is required when --layout plugin is used[/red]")
        raise typer.Exit(1)

    if layout == "standalone":
        skill_dir = output / skill_name
    else:
        skill_dir = output / "plugins" / str(plugin) / "skills" / skill_name
    evals_dir = skill_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)

    example_evals, files = _init_template_payload(skill_name, template)

    evals_path = evals_dir / "evals.json"
    with open(evals_path, "w", encoding="utf-8") as f:
        json.dump(example_evals, f, indent=2)

    (evals_dir / "files").mkdir(exist_ok=True)
    for rel_path, content in files.items():
        path = evals_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    skill_md_path = skill_dir / "SKILL.md"
    if not skill_md_path.exists():
        skill_md_path.write_text(SKILL_MD_TEMPLATE.format(skill_name=skill_name))

    console.print(f"[green]Created {template} eval structure at {evals_dir}[/green]")
    console.print(f"  - {evals_path}")
    console.print(f"  - {evals_dir / 'files'}/ (add fixture files here)")
    console.print(f"  - {skill_md_path} (skill template; edit before evaluating)")


if __name__ == "__main__":
    app()
