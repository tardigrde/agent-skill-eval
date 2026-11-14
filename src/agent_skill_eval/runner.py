from __future__ import annotations

import json
import os
import statistics
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from agent_skill_eval import __version__
from agent_skill_eval.fixtures import build_fixture_file_map
from agent_skill_eval.git_state import capture_git_state, github_repo_slug, repo_project_path, repo_provider
from agent_skill_eval.graders import DEFAULT_GRADER_MODEL, LLMGrader, grade_assertions, summarize_assertion_results
from agent_skill_eval.harnesses import HARNESSES, detect_skill_invocation, get_cli_version, get_harness
from agent_skill_eval.hooks import apply_post_grade_hooks, run_suite_hooks
from agent_skill_eval.models import (
    AgentType,
    AssertionResult,
    BenchmarkResult,
    BenchmarkStats,
    BudgetConfig,
    CleanupManifest,
    DeltaStats,
    EvalCase,
    EvalSuite,
    GradingResult,
    RunMeta,
    StatsPair,
    TimingData,
)
from agent_skill_eval.progress import ProgressLog, format_tokens, run_label
from agent_skill_eval.skills import SkillInstaller
from agent_skill_eval.summary import build_run_summary, write_run_summary
from agent_skill_eval.validation import EvalSuiteValidationReport, load_and_validate_suite
from agent_skill_eval.workspace import WorkspaceManager

console = Console()


class EvalRunner:
    def __init__(
        self,
        skill_path: Path,
        evals_path: Path,
        workspace_base: Path,
        agents: list[AgentType],
        concurrency: int = 1,
        with_baseline: bool = True,
        grader_model: str = DEFAULT_GRADER_MODEL,
        grader_base_url: Optional[str] = None,
        grader_key_file: Optional[Path] = None,
        grader_allowed_models: Optional[tuple[str, ...]] = None,
        agent_models: Optional[dict[AgentType, str]] = None,
        source_repo: Optional[str] = None,
        harness_base_url: Optional[str] = None,
        agent_timeout: Optional[int] = None,
        agent_max_retries: Optional[int] = None,
        runs: int = 1,
        eval_ids: Optional[list[str]] = None,
        side_effect_levels: Optional[list[str]] = None,
        budget: Optional[BudgetConfig] = None,
        pricing: Optional[dict[str, dict]] = None,
        pre_run_commands: Optional[list[str]] = None,
        post_grade_commands: Optional[list[str]] = None,
        post_run_commands: Optional[list[str]] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.skill_path = skill_path
        self.evals_path = evals_path
        self.workspace_base = workspace_base
        self.agents = agents
        self.concurrency = concurrency
        self.with_baseline = with_baseline
        self.grader_model = grader_model
        self.grader_base_url = grader_base_url
        self.grader_key_file = grader_key_file
        self.grader_allowed_models = grader_allowed_models
        self.agent_models = agent_models or {}
        self.source_repo = source_repo
        self.harness_base_url = harness_base_url
        self.agent_timeout = agent_timeout
        self.agent_max_retries = agent_max_retries
        self.runs = max(1, runs)
        self.eval_ids = [str(e) for e in eval_ids] if eval_ids else []
        self.side_effect_levels = [str(level) for level in side_effect_levels] if side_effect_levels else []
        self.budget = budget if budget and budget.enabled else None
        self.pricing = pricing or {}
        self.pre_run_commands = pre_run_commands or []
        self.post_grade_commands = post_grade_commands or []
        self.post_run_commands = post_run_commands or []
        self.reasoning_effort = reasoning_effort

        self.run_id = uuid.uuid4().hex[:8]
        # One version probe per agent for the whole suite; recorded in every
        # run_meta.json so results are attributable to an exact CLI build.
        self._cli_versions: dict[AgentType, Optional[str]] = {a: get_cli_version(a) for a in self.agents}
        self.skill_installer = SkillInstaller(skill_path)
        self.skill_hash = self.skill_installer.content_hash()
        self.skill_size = self.skill_installer.size_metrics()
        self.suite = self._load_suite()
        self.validation_report: EvalSuiteValidationReport
        self.workspace_mgr = WorkspaceManager(workspace_base, source_repo=source_repo)
        self.llm_grader = (
            LLMGrader(
                model=grader_model,
                base_url=grader_base_url,
                api_key_file=grader_key_file,
                allowed_models=grader_allowed_models,
            )
            if grader_model
            else None
        )

        # Set when a budget violation with action=stop-suite fires; runs
        # that have not started yet short-circuit to "skipped".
        self._stop_suite = threading.Event()
        self._stop_reason: Optional[str] = None
        self._progress: Optional[ProgressLog] = None

    def _load_suite(self) -> EvalSuite:
        # Filter once, before tasks or metadata exist, so every downstream
        # artifact (evals_meta.json, workspaces, benchmark) naturally
        # describes only the selected cases.
        suite, report = load_and_validate_suite(self.evals_path, self.eval_ids, self.side_effect_levels)
        self.validation_report = report
        return suite

    def run(self, iteration: int = 1) -> Path:
        started = time.time()
        iteration_dir = self.workspace_base / f"{self.suite.skill_name}-workspace" / f"iteration-{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        self._save_eval_metadata(iteration_dir)
        self._progress = ProgressLog(iteration_dir / "progress.jsonl")

        if self.reasoning_effort:
            unsupported = [a.value for a in self.agents if not HARNESSES[a].supports_reasoning_effort]
            if unsupported:
                console.print(
                    f"[yellow]Warning: --reasoning-effort is not supported by {', '.join(unsupported)}; "
                    f"those agents run on their own local defaults (run_meta.json records null)[/yellow]"
                )

        hook_env = self._suite_hook_env(iteration_dir, iteration)
        hook_records: dict[str, list[dict]] = {"pre_run": [], "post_run": []}
        if self.pre_run_commands:
            console.print(f"[cyan]Running {len(self.pre_run_commands)} pre-run hook(s)...[/cyan]")
            # A failing pre-run hook raises HookError here, before any
            # model call or workspace mutation.
            hook_records["pre_run"] = run_suite_hooks(self.pre_run_commands, hook_env, label="pre-run", fail_fast=True)

        tasks = []
        for eval_case in self.suite.evals:
            for agent_type in self.agents:
                for run_index in range(1, self.runs + 1):
                    tasks.append((eval_case, agent_type, True, run_index))
                    if self.with_baseline:
                        tasks.append((eval_case, agent_type, False, run_index))

        self._progress.emit(
            "suite_started",
            skill=self.suite.skill_name,
            iteration=iteration,
            run_id=self.run_id,
            total_runs=len(tasks),
            eval_ids=[str(e.id) for e in self.suite.evals],
            agents=[a.value for a in self.agents],
            skill_hash=self.skill_hash,
            side_effect_levels=sorted({e.side_effect_level for e in self.suite.evals}),
        )

        results: dict[str, dict] = {}
        cleanup_entries: list[CleanupManifest] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Running evals...", total=len(tasks))

            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {}
                for eval_case, agent_type, with_skill, run_index in tasks:
                    future = executor.submit(
                        self._run_single,
                        eval_case,
                        agent_type,
                        with_skill,
                        iteration_dir,
                        iteration,
                        run_index,
                    )
                    futures[future] = (eval_case, agent_type, with_skill, run_index)

                # Inside the executor block on purpose: collecting results
                # as they complete keeps the progress bar and progress.jsonl
                # live instead of only updating after every run finished.
                for future in as_completed(futures):
                    eval_case, agent_type, with_skill, run_index = futures[future]
                    key = f"{eval_case.id}-{agent_type.value}-{'with' if with_skill else 'without'}-r{run_index}"
                    try:
                        result = future.result()
                        results[key] = result["summary"]
                        if result.get("cleanup_entry"):
                            cleanup_entries.append(result["cleanup_entry"])
                    except Exception as e:
                        console.print(f"[red]Error running {key}: {e}[/red]")
                        results[key] = {"error": str(e)}
                        self._progress.emit(
                            "run_error",
                            eval_id=str(eval_case.id),
                            agent=agent_type.value,
                            with_skill=with_skill,
                            run_index=run_index,
                            error=str(e),
                        )
                    progress.advance(task_id)

        benchmark = self._compute_benchmark(results, iteration_dir)

        if self.with_baseline:
            for agent_type in self.agents:
                if agent_type.value not in benchmark.deltas:
                    console.print(
                        f"[yellow]Warning: no with/without-skill delta for {agent_type.value} — "
                        f"all of its runs in one config errored[/yellow]"
                    )

        with open(iteration_dir / "benchmark.json", "w", encoding="utf-8") as f:
            json.dump(benchmark.model_dump(), f, indent=2)

        manifest = self._build_cleanup_manifest(cleanup_entries, iteration_dir)
        with open(iteration_dir / "cleanup.json", "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(exclude_none=True), f, indent=2)

        if self.post_run_commands:
            console.print(f"[cyan]Running {len(self.post_run_commands)} post-run hook(s)...[/cyan]")
            hook_records["post_run"] = run_suite_hooks(
                self.post_run_commands, hook_env, label="post-run", fail_fast=False
            )
            for record in hook_records["post_run"]:
                if record["exit_code"] != 0:
                    console.print(
                        f"[yellow]post-run hook failed (exit {record['exit_code']}): {record['command']}[/yellow]"
                    )

        summary = build_run_summary(
            skill_name=self.suite.skill_name,
            iteration=iteration,
            run_id=self.run_id,
            agents=[a.value for a in self.agents],
            eval_ids=[str(e.id) for e in self.suite.evals],
            side_effect_levels=sorted({e.side_effect_level for e in self.suite.evals}),
            runs=self.runs,
            with_baseline=self.with_baseline,
            duration_seconds=time.time() - started,
            results=results,
            benchmark=benchmark.model_dump(),
            cleanup=manifest.model_dump(exclude_none=True),
            skill_hash=self.skill_hash,
            skill_size=self.skill_size,
            budget=self.budget.model_dump() if self.budget else None,
            hooks=hook_records,
        )
        write_run_summary(iteration_dir, summary)

        self._progress.emit(
            "suite_finished",
            total_runs=len(tasks),
            completed=summary["runs_completed"],
            failed_grading=summary["runs_failed_grading"],
            errors=len(summary["errors"]),
            skipped=len(summary["skipped_runs"]),
        )

        if self._stop_suite.is_set():
            console.print(f"[yellow]Suite stopped early by budget: {self._stop_reason}[/yellow]")

        console.print(f"\n[green]Results saved to {iteration_dir}[/green]")
        return iteration_dir

    def _save_eval_metadata(self, iteration_dir: Path) -> None:
        meta = {
            "skill_name": self.suite.skill_name,
            "skill_hash": self.skill_hash,
            "skill_size": self.skill_size,
            "evals": [eval_case.model_dump() for eval_case in self.suite.evals],
            "source_repo": self.source_repo,
            # The case filter actually applied (--eval-id); empty means the
            # full suite ran. report/compare readers can tell filtered runs
            # apart without diffing the evals list.
            "selected_eval_ids": self.eval_ids,
            "selected_side_effect_levels": self.side_effect_levels,
            "side_effect_counts": self.validation_report.side_effect_counts,
            "grader": {
                "model": self.grader_model,
                "base_url": self.grader_base_url,
                "key_file": str(self.grader_key_file) if self.grader_key_file else None,
                "allowed_models": list(self.grader_allowed_models) if self.grader_allowed_models else None,
            },
        }
        with open(iteration_dir / "evals_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _suite_hook_env(self, iteration_dir: Path, iteration: int) -> dict[str, str]:
        return {
            "ASE_SKILL_NAME": self.suite.skill_name,
            "ASE_SKILL_HASH": self.skill_hash,
            "ASE_SKILL_DIR": str(self.skill_path),
            "ASE_EVALS_PATH": str(self.evals_path),
            "ASE_ITERATION": str(iteration),
            "ASE_ITERATION_DIR": str(iteration_dir),
            "ASE_WORKSPACE_BASE": str(self.workspace_base),
            "ASE_SOURCE_REPO": self.source_repo or "",
            "ASE_RUN_ID": self.run_id,
            "ASE_EVAL_IDS": ",".join(str(e.id) for e in self.suite.evals),
            "ASE_SIDE_EFFECT_LEVELS": ",".join(sorted({e.side_effect_level for e in self.suite.evals})),
        }

    def _build_cleanup_manifest(self, entries: list[CleanupManifest], iteration_dir: Path) -> CleanupManifest:
        all_branches: list[str] = []
        all_review_requests: list[int] = []
        for entry in entries:
            for b in entry.remote_branches:
                if b not in all_branches:
                    all_branches.append(b)
            for p in entry.pr_numbers:
                if p not in all_review_requests:
                    all_review_requests.append(p)
            for mr in entry.merge_request_numbers:
                if mr not in all_review_requests:
                    all_review_requests.append(mr)

        provider = repo_provider(self.source_repo)

        return CleanupManifest(
            source_repo=self.source_repo,
            source_repo_provider=provider,
            source_repo_project=repo_project_path(self.source_repo),
            source_repo_slug=github_repo_slug(self.source_repo),
            remote_branches=all_branches,
            pr_numbers=[] if provider == "gitlab" else all_review_requests,
            merge_request_numbers=all_review_requests if provider == "gitlab" else [],
            workspaces=[str(p) for p in self.workspace_mgr.workspaces],
        )

    def _emit(self, event: str, **fields) -> None:
        if self._progress is not None:
            self._progress.emit(event, **fields)

    def _run_single(
        self,
        eval_case: EvalCase,
        agent_type: AgentType,
        with_skill: bool,
        iteration_dir: Path,
        iteration: int = 1,
        run_index: int = 1,
    ) -> dict:
        label = run_label(eval_case.id, agent_type.value, with_skill, run_index, self.runs)
        identity = {
            "eval_id": str(eval_case.id),
            "agent": agent_type.value,
            "with_skill": with_skill,
            "run_index": run_index,
        }

        if self._stop_suite.is_set():
            reason = f"skipped: {self._stop_reason or 'suite stopped by budget'}"
            console.print(f"  [yellow]{escape(f'[{label}]')} {reason}[/yellow]")
            self._emit("run_skipped", **identity, reason=reason)
            return {
                "summary": {
                    "eval_id": eval_case.id,
                    "agent": agent_type.value,
                    "with_skill": with_skill,
                    "run_index": run_index,
                    "error": reason,
                    "skipped": True,
                },
                "cleanup_entry": None,
            }

        run_started = time.time()
        trajectory: list[dict] = []

        def record(event: str, **fields) -> None:
            trajectory.append({"event": event, "elapsed_ms": int((time.time() - run_started) * 1000), **fields})

        console.print(f"  [dim]{escape(f'[{label}]')} started[/dim]")
        self._emit("run_started", **identity)
        record("run_started", **identity, side_effect_level=eval_case.side_effect_level)

        eval_slug = f"eval-{eval_case.id}"
        config_slug = "with_skill" if with_skill else "without_skill"
        agent_slug = agent_type.value
        config_dir = iteration_dir / eval_slug / agent_slug / config_slug
        if self.runs > 1:
            config_dir = config_dir / f"run-{run_index}"
        output_dir = config_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        fixture_files = build_fixture_file_map(self.evals_path.parent, eval_case.files)

        workspace_name = f"{eval_case.id}-{agent_type.value}-{'ws' if with_skill else 'bs'}"
        if self.runs > 1:
            workspace_name += f"-r{run_index}"
        workspace = self.workspace_mgr.create_workspace(workspace_name, fixture_files)
        record("workspace_created", path=str(workspace), fixture_count=len(fixture_files))

        if with_skill:
            installed_to = self.skill_installer.install(workspace, agent_type)
            # The skill is eval input, not an agent edit: commit it into the
            # baseline so git_clean, file_unchanged and side-effect contracts
            # only see changes the agent actually made.
            committed = self.workspace_mgr.commit_paths(
                workspace,
                self.skill_installer.install_targets(workspace, agent_type),
                f"Install skill {self.suite.skill_name}",
            )
            record(
                "skill_installed",
                path=str(installed_to),
                skill_hash=self.skill_hash,
                baseline_commit=committed,
            )

        if eval_case.stage_files and fixture_files:
            for name in fixture_files:
                subprocess.run(
                    ["git", "add", "--", name],
                    cwd=workspace,
                    capture_output=True,
                    check=False,
                )
            record("fixtures_staged", count=len(fixture_files))

        model = self.agent_models.get(agent_type)
        harness = get_harness(
            agent_type,
            workspace,
            model,
            base_url=self.harness_base_url,
            timeout=self.agent_timeout,
            max_retries=self.agent_max_retries,
            pricing=self.pricing,
            reasoning_effort=self.reasoning_effort,
        )

        prompt = self._build_prompt(eval_case, with_skill)

        self._emit("snapshot_started", **identity, phase="pre")
        pre_state = capture_git_state(workspace, source_repo=self.source_repo)
        record("git_state_captured", phase="pre", current_branch=pre_state.current_branch, head_sha=pre_state.head_sha)
        with open(output_dir / "pre_state.json", "w", encoding="utf-8") as f:
            json.dump(pre_state.model_dump(), f, indent=2)

        self._emit("agent_started", **identity)
        record("agent_started", model=model)
        agent_output, timing, stdout, stderr = harness.run(prompt, output_dir)
        installed_paths = self.skill_installer.install_targets(workspace, agent_type) if with_skill else []
        skill_invoked, skill_invocation_evidence = detect_skill_invocation(
            agent_type, stdout, stderr, self.suite.skill_name, installed_paths
        )
        record(
            "agent_finished",
            exit_code=timing.exit_code,
            timed_out=timing.timed_out,
            duration_ms=timing.duration_ms,
            total_tokens=timing.total_tokens,
            cost_usd=timing.cost_usd,
        )
        record(
            "skill_invocation_checked",
            invoked=skill_invoked,
            evidence=skill_invocation_evidence,
        )
        self._emit(
            "agent_finished",
            **identity,
            duration_s=round(timing.duration_ms / 1000.0, 1),
            input_tokens=timing.input_tokens,
            cached_tokens=timing.cached_tokens,
            non_cached_input_tokens=timing.non_cached_input_tokens,
            output_tokens=timing.output_tokens,
            reasoning_output_tokens=timing.reasoning_output_tokens,
            total_tokens=timing.total_tokens,
            cost_usd=timing.cost_usd,
            exit_code=timing.exit_code,
            timed_out=timing.timed_out,
        )

        budget_violations = self._check_budget(timing, label)
        if budget_violations:
            record("budget_exceeded", violations=budget_violations)

        self._emit("snapshot_started", **identity, phase="post")
        post_state = capture_git_state(workspace, source_repo=self.source_repo)
        record(
            "git_state_captured",
            phase="post",
            current_branch=post_state.current_branch,
            head_sha=post_state.head_sha,
        )
        with open(output_dir / "post_state.json", "w", encoding="utf-8") as f:
            json.dump(post_state.model_dump(), f, indent=2)

        with open(output_dir / "output.txt", "w", encoding="utf-8") as f:
            f.write(agent_output)

        timing_path = output_dir.parent / "timing.json"
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(timing.model_dump(), f, indent=2)

        run_meta = RunMeta(
            eval_id=eval_case.id,
            agent=agent_type.value,
            with_skill=with_skill,
            iteration=iteration,
            skill_name=self.suite.skill_name,
            skill_hash=self.skill_hash,
            side_effect_level=eval_case.side_effect_level,
            source_repo=self.source_repo,
            run_id=self.run_id,
            run_index=run_index,
            model=model,
            # The value the harness actually applied: None for agents that
            # cannot pass it through (they run on their own local defaults).
            reasoning_effort=(self.reasoning_effort if HARNESSES[agent_type].supports_reasoning_effort else None),
            base_url=self.harness_base_url,
            harness_version=__version__,
            agent_cli_version=self._cli_versions.get(agent_type),
            skill_invoked=skill_invoked,
            skill_invocation_evidence=skill_invocation_evidence,
        )
        meta_path = output_dir.parent / "run_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(run_meta.model_dump(), f, indent=2)

        grading = grade_assertions(
            eval_case.assertions,
            agent_output,
            output_dir,
            eval_case.expected_output,
            self.llm_grader,
            workspace=workspace,
            pre_state=pre_state,
            post_state=post_state,
            should_trigger=eval_case.should_trigger,
        )
        grading = self._apply_side_effect_contract(grading, eval_case, pre_state, post_state)
        record(
            "graded",
            passed=grading.summary.passed,
            failed=grading.summary.failed,
            skipped=grading.summary.skipped,
            total=grading.summary.total,
            pass_rate=grading.summary.pass_rate,
        )

        grading_path = output_dir.parent / "grading.json"
        with open(grading_path, "w", encoding="utf-8") as f:
            json.dump(grading.model_dump(), f, indent=2)

        if self.post_grade_commands:
            hook_env = self._suite_hook_env(iteration_dir, iteration)
            hook_env.update(
                {
                    "ASE_EVAL_ID": str(eval_case.id),
                    "ASE_AGENT": agent_type.value,
                    "ASE_WITH_SKILL": "1" if with_skill else "0",
                    "ASE_RUN_INDEX": str(run_index),
                    "ASE_WORKSPACE_PATH": str(workspace),
                    "ASE_OUTPUT_DIR": str(output_dir),
                    "ASE_PRE_STATE_PATH": str(output_dir / "pre_state.json"),
                    "ASE_POST_STATE_PATH": str(output_dir / "post_state.json"),
                    "ASE_TIMING_PATH": str(timing_path),
                    "ASE_RUN_META_PATH": str(meta_path),
                    "ASE_GRADING_PATH": str(grading_path),
                }
            )
            grading, hook_records = apply_post_grade_hooks(self.post_grade_commands, hook_env, grading)
            with open(output_dir / "post_grade_hooks.json", "w", encoding="utf-8") as f:
                json.dump(hook_records, f, indent=2)
            record("post_grade_hooks_finished", count=len(hook_records))

        grading = self._apply_budget_verdict(grading, budget_violations)
        if budget_violations and self.budget and self.budget.action != "warn":
            record("budget_verdict_applied", action=self.budget.action)

        # Rewritten when hooks or budget added results above; cheap either way.
        with open(grading_path, "w", encoding="utf-8") as f:
            json.dump(grading.model_dump(), f, indent=2)

        cleanup_entry = self._build_cleanup_entry(pre_state, post_state)
        record(
            "cleanup_entry_built",
            remote_branches=cleanup_entry.remote_branches,
            pr_numbers=cleanup_entry.pr_numbers,
        )

        if os.environ.get("ASE_KEEP_WORKSPACE"):
            console.print(f"  [dim]Workspace kept: {workspace}[/dim]")
            record("workspace_kept", path=str(workspace))
        else:
            self.workspace_mgr.cleanup(workspace)
            record("workspace_cleaned", path=str(workspace))

        with open(output_dir.parent / "trajectory.json", "w", encoding="utf-8") as f:
            json.dump(trajectory, f, indent=2)

        cost_str = f"{timing.cost_usd:.4f} USD" if timing.cost_usd is not None else "n/a"
        reasoning_str = (
            f" reasoning={format_tokens(timing.reasoning_output_tokens)}"
            if timing.reasoning_output_tokens is not None
            else ""
        )
        console.print(
            f"  [dim]{escape(f'[{label}]')} finished in {time.time() - run_started:.0f}s: "
            f"{grading.summary.passed}/{grading.summary.total} passed, "
            f"tokens_in={format_tokens(timing.input_tokens)} "
            f"cached={format_tokens(timing.cached_tokens)} "
            f"out={format_tokens(timing.output_tokens)}{reasoning_str} cost={cost_str}[/dim]"
        )
        self._emit(
            "run_finished",
            **identity,
            duration_s=round(time.time() - run_started, 1),
            pass_rate=grading.summary.pass_rate,
            passed=grading.summary.passed,
            total=grading.summary.total,
            total_tokens=timing.total_tokens,
            budget_exceeded=timing.budget_exceeded,
        )

        return {
            "summary": {
                "eval_id": eval_case.id,
                "agent": agent_type.value,
                "with_skill": with_skill,
                "run_index": run_index,
                "side_effect_level": eval_case.side_effect_level,
                "timing": timing.model_dump(),
                "grading": grading.model_dump(),
                "agent_output": agent_output[:2000],
                "skill_invoked": skill_invoked,
                "skill_invocation_evidence": skill_invocation_evidence,
            },
            "cleanup_entry": cleanup_entry,
        }

    def _check_budget(self, timing: TimingData, label: str) -> list[str]:
        """Evaluate per-case budgets against a finished run's timing.

        Enforcement is post-run: the agent process has already exited (a
        case that must be killed mid-run is the timeout's job). Records the
        verdict on the timing object; "stop-suite" additionally prevents
        any not-yet-started run from launching.
        """
        if not self.budget:
            return []
        violations = self.budget.violations(timing)
        if not violations:
            return []

        reason = "; ".join(violations)
        timing.budget_exceeded = True
        timing.budget_reason = reason
        console.print(f"  [yellow]{escape(f'[{label}]')} budget exceeded ({self.budget.action}): {reason}[/yellow]")
        self._emit("budget_exceeded", label=label, reason=reason, action=self.budget.action)

        if self.budget.action == "stop-suite" and not self._stop_suite.is_set():
            self._stop_reason = f"budget exceeded by [{label}]: {reason}"
            self._stop_suite.set()
        return violations

    def _apply_budget_verdict(self, grading: GradingResult, violations: list[str]) -> GradingResult:
        """Fail the run's grading when its budget was exceeded (action != warn)."""
        if not violations or not self.budget or self.budget.action == "warn":
            return grading
        all_results = list(grading.assertion_results) + [
            AssertionResult(
                text="The run stayed within its budget",
                passed=False,
                evidence="budget exceeded: " + "; ".join(violations),
                method="budget",
            )
        ]
        return GradingResult(
            assertion_results=all_results,
            summary=summarize_assertion_results(all_results),
        )

    def _apply_side_effect_contract(
        self,
        grading: GradingResult,
        eval_case: EvalCase,
        pre_state,
        post_state,
    ) -> GradingResult:
        contract = eval_case.side_effect_contract
        if contract is None:
            return grading

        problems: list[str] = []
        new_local = sorted(set(post_state.local_branches) - set(pre_state.local_branches))
        new_remote = sorted(set(post_state.remote_branches) - set(pre_state.remote_branches))
        new_commits = sorted(set(post_state.commit_shas) - set(pre_state.commit_shas))
        pre_reviews = {item.get("number") for item in pre_state.open_prs}
        new_reviews = sorted(
            str(item.get("number"))
            for item in post_state.open_prs
            if item.get("number") is not None and item.get("number") not in pre_reviews
        )
        changed_files = sorted(set(post_state.changed_files))

        if new_local and not contract.allow_new_local_branches:
            problems.append("new local branches: " + ", ".join(new_local))
        if new_remote and not contract.allow_new_remote_branches:
            problems.append("new remote branches: " + ", ".join(new_remote))
        if new_commits and not contract.allow_new_commits:
            problems.append(f"new commits: {len(new_commits)}")
        if new_reviews and not contract.allow_new_review_requests:
            problems.append("new review requests: " + ", ".join(new_reviews))
        if changed_files and not contract.allow_worktree_changes:
            problems.append("worktree changes: " + ", ".join(changed_files))

        result = AssertionResult(
            text="The run respected its side-effect contract",
            passed=not problems,
            evidence="ok" if not problems else "; ".join(problems),
            method="side-effect-contract",
        )
        all_results = list(grading.assertion_results) + [result]
        return GradingResult(assertion_results=all_results, summary=summarize_assertion_results(all_results))

    def _build_prompt(self, eval_case: EvalCase, with_skill: bool) -> str:
        """Build the prompt sent to the agent.

        With-skill mode means the skill is installed and available, NOT that
        the prompt is rewritten to invoke it. Eval prompts decide for
        themselves whether to mention the skill, except for the explicit
        ``force_skill_invocation`` opt-in (used by tests that need to assert
        the skill is reachable by name).
        """
        if with_skill and eval_case.force_skill_invocation:
            return f"Use the ${self.suite.skill_name} skill. {eval_case.prompt}"
        return eval_case.prompt

    def _build_cleanup_entry(
        self,
        pre_state,
        post_state,
    ) -> CleanupManifest:
        pre_remote = set(pre_state.remote_branches)
        post_remote = set(post_state.remote_branches)
        new_remote_branches = sorted(post_remote - pre_remote)

        pre_pr_numbers = {p.get("number") for p in pre_state.open_prs}
        new_pr_numbers: list[int] = []
        for p in post_state.open_prs:
            number = p.get("number")
            if number is None or number in pre_pr_numbers:
                continue
            # The PR snapshot is taken with --state all: a PR merged or
            # closed during the run needs no cleanup.
            state = (p.get("state") or "OPEN").upper()
            if state != "OPEN":
                continue
            # A PR someone else opened on the source repo mid-run is new in
            # the snapshot but not ours: only record PRs whose head branch
            # this run pushed.
            head = p.get("headRefName")
            if head is not None and head not in new_remote_branches:
                continue
            new_pr_numbers.append(number)

        provider = repo_provider(self.source_repo)

        return CleanupManifest(
            source_repo=self.source_repo,
            source_repo_provider=provider,
            source_repo_project=repo_project_path(self.source_repo),
            source_repo_slug=github_repo_slug(self.source_repo),
            remote_branches=new_remote_branches,
            pr_numbers=[] if provider == "gitlab" else new_pr_numbers,
            merge_request_numbers=new_pr_numbers if provider == "gitlab" else [],
        )

    def _compute_benchmark(self, results: dict[str, dict], iteration_dir: Path) -> BenchmarkResult:
        return compute_benchmark(results, self.agents, self.with_baseline)

    def _compute_stats(self, results: list[dict]) -> BenchmarkStats:
        return compute_stats(results)


def _is_full_pass(result: dict) -> bool:
    summary = result["grading"]["summary"]
    return summary.get("total", 1) > 0 and summary.get("pass_rate", 0.0) >= 1.0


def compute_stats(results: list[dict]) -> BenchmarkStats:
    if not results:
        return BenchmarkStats(
            pass_rate=StatsPair(mean=0.0, stddev=0.0),
            time_seconds=StatsPair(mean=0.0, stddev=0.0),
            tokens=StatsPair(mean=0.0, stddev=0.0),
        )

    pass_rates = [r["grading"]["summary"]["pass_rate"] for r in results]
    times = [r["timing"]["duration_ms"] / 1000.0 for r in results]
    tokens = [r["timing"]["total_tokens"] for r in results]
    # Only runs that reported a cost contribute; None means unavailable
    # (codex without --pricing-config), which must not average in as $0.
    # .get: timing.json from runs before cost_usd existed lacks the field.
    costs = [c for r in results if (c := r["timing"].get("cost_usd")) is not None]

    full_passes = [_is_full_pass(r) for r in results]

    # pass@k: group runs by eval id; an eval counts as passed if ANY of its
    # runs fully passed. With one run per eval this equals full_pass_rate.
    by_eval: dict[str, list[bool]] = {}
    for r, fp in zip(results, full_passes):
        by_eval.setdefault(str(r.get("eval_id", "?")), []).append(fp)
    k = max(len(v) for v in by_eval.values())
    pass_at_k = sum(1 for v in by_eval.values() if any(v)) / len(by_eval)

    return BenchmarkStats(
        pass_rate=StatsPair(
            mean=statistics.mean(pass_rates) if pass_rates else 0.0,
            stddev=statistics.stdev(pass_rates) if len(pass_rates) > 1 else 0.0,
        ),
        time_seconds=StatsPair(
            mean=statistics.mean(times) if times else 0.0,
            stddev=statistics.stdev(times) if len(times) > 1 else 0.0,
        ),
        tokens=StatsPair(
            mean=statistics.mean(tokens) if tokens else 0.0,
            stddev=statistics.stdev(tokens) if len(tokens) > 1 else 0.0,
        ),
        cost_usd=StatsPair(
            mean=statistics.mean(costs),
            stddev=statistics.stdev(costs) if len(costs) > 1 else 0.0,
        )
        if costs
        else None,
        cost_runs=len(costs),
        full_pass_rate=sum(full_passes) / len(full_passes),
        pass_at_k=pass_at_k,
        k=k,
    )


def compute_benchmark(results: dict[str, dict], agents: list[AgentType], with_baseline: bool) -> BenchmarkResult:
    """Compute the benchmark summary from a results dict.

    Module-level so it can be called from the ``grade`` command without a
    full ``EvalRunner`` instance. Computes a with/without-skill delta for
    EVERY agent (``deltas``); ``delta`` is kept for backward compatibility
    and is only set when exactly one agent was run.
    """
    run_summary = {}
    deltas: dict[str, DeltaStats] = {}

    for agent_type in agents:
        with_results = [
            r
            for k, r in results.items()
            if r.get("agent") == agent_type.value and r.get("with_skill") and "error" not in r
        ]
        without_results = [
            r
            for k, r in results.items()
            if r.get("agent") == agent_type.value and not r.get("with_skill") and "error" not in r
        ]

        with_stats = compute_stats(with_results)
        without_stats = compute_stats(without_results)

        run_summary[f"{agent_type.value}_with_skill"] = with_stats
        if with_baseline:
            run_summary[f"{agent_type.value}_without_skill"] = without_stats
            if with_results and without_results:
                cost_delta = None
                if with_stats.cost_usd is not None and without_stats.cost_usd is not None:
                    cost_delta = with_stats.cost_usd.mean - without_stats.cost_usd.mean
                deltas[agent_type.value] = DeltaStats(
                    pass_rate=with_stats.pass_rate.mean - without_stats.pass_rate.mean,
                    time_seconds=with_stats.time_seconds.mean - without_stats.time_seconds.mean,
                    tokens=with_stats.tokens.mean - without_stats.tokens.mean,
                    cost_usd=cost_delta,
                )

    delta = None
    if len(agents) == 1 and agents[0].value in deltas:
        delta = deltas[agents[0].value]

    return BenchmarkResult(run_summary=run_summary, deltas=deltas, delta=delta)
