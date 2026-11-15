# agent-skill-eval

**Evaluate agent skills through the real coding harnesses you use every day — Claude Code, Codex, and OpenCode — not the raw API.**

You wrote a `SKILL.md`. Does it actually make your agent better? agent-skill-eval answers that with data: it installs your skill into a fresh workspace, runs the *actual agent CLI* against your test prompts (with and without the skill), grades the results with deterministic state-diff checks plus an LLM rubric, and reports the measured impact — pass rates, pass^k across repeated runs, token costs, and wall-clock time.

Runs in one of two backends: **Harbor** (recommended) puts every trial in its own container with a separate verifier and a network policy; the **native** runner uses a host workspace for fast, unisolated iteration.

Because the eval goes through the full harness — system prompt, skill discovery, permissions, tool use — you get exactly the behavior you'll see in daily use, including the failure mode that matters most: *the agent never triggering your skill at all*.

## Features

- **Real harnesses, end to end**: OpenCode, Claude Code, and Codex CLIs, in a single run
- **Isolated Harbor backend**: one container per trial, a separate verifier, and a network policy — the agent cannot read the grader, sibling workspaces, or the skill from the internet
- **Baseline comparison**: with-skill vs. without-skill, with a per-agent delta
- **pass^k and pass@k**: `--runs N` repeats every eval and reports both metrics with Wilson intervals, because agents are stochastic and single-run numbers lie
- **Cost and time next to correctness**: median and p90 tokens, cost, and wall time per arm, so a change that keeps correctness but doubles cost shows as a regression
- **Trigger split**: positive cases and negative controls are reported separately; unknown telemetry is never counted as "did not trigger"
- **State-delta grading**: code-based checks compare pre/post git state snapshots, so they don't false-pass on pre-existing branches, commits, or review requests
- **Negative controls**: `should_trigger: false` inverts assertions to catch accidental skill triggering
- **Honest grading**: assertions the grader can't check are *skipped*, not failed; a missing API key warns upfront instead of silently zeroing your pass rate
- **Pinned models**: `--agent-model claude-code=haiku --agent-model codex=gpt-5-mini` makes runs reproducible across machines
- **Recorded effective config**: every run's `run_meta.json` records model, reasoning effort, base URL, agent CLI version, and harness version — so "codex passed 97%" is attributable to an exact configuration, not whatever local config the CLI silently inherited
- **Reasoning control and telemetry**: `--reasoning-effort` pins the reasoning setting for agents that support it (codex), and reasoning output tokens are persisted in `timing.json`/`summary.json` when the agent reports them
- **Targeted reruns**: `--eval-id` runs/validates a single case instead of the whole suite — the cheap inner loop after a one-line skill edit
- **Budget guards**: per-case token/cost/duration limits with `warn`, `fail`, or `stop-suite` actions, so a runaway recovery loop can't silently burn quota
- **Live progress**: per-case console lines plus a tailable `progress.jsonl` — long runs are no longer opaque
- **Lifecycle hooks**: `--pre-run-command`, `--post-grade-command`, `--post-run-command` plug external setup, side-effect graders, and teardown into the run with structured `ASE_*` metadata
- **One-file summaries**: every run writes `summary.json` (pass rates, failures, token/cost totals, cleanup state); `agent-skill-eval status` reads it back without a model call
- **Scoped cleanup**: only removes artifacts recorded in `cleanup.json`; never closes unrelated PRs/MRs or deletes unrelated branches
- **Re-grading**: `agent-skill-eval grade` re-grades saved outputs without re-running agents
- **Markdown reports**: paste `agent-skill-eval report --format markdown` straight into a PR or blog post

## Installation

```bash
pip install agent-skill-eval

# Recommended: the isolated Harbor backend (Python 3.12+, Docker)
pip install 'agent-skill-eval[harbor]'
```

The native runner supports Python 3.11+. Harbor 0.26 requires Python 3.12+ and Docker. The `harbor` extra uses a version marker, so installing it on 3.11 is a no-op and `--backend harbor` reports a clear error.

This installs two identical commands: `agent-skill-eval` and the short alias `ase`. The CLI is also runnable as a module — `python -m agent_skill_eval run ...` — which is handy when the scripts directory isn't on your `PATH` or you want to pin the interpreter (e.g. `uv run python -m agent_skill_eval`).

> **Coming soon:** subagent evals — evaluate custom subagent definitions the same way as skills, across the same harnesses.

Or from source with `uv`:

```bash
git clone https://github.com/tardigrde/agent-skill-eval
cd agent-skill-eval
uv venv && uv pip install -e ".[dev]"
```

You also need the agent CLIs you want to evaluate (`claude`, `codex`, `opencode`) installed and authenticated, and an `OPENROUTER_API_KEY` or `OPENAI_API_KEY` for LLM rubric grading. The Harbor backend installs the agent CLI inside the task container, so the host only needs Docker and the model API key.

## Backends

| Backend | Isolation | Best for |
| --- | --- | --- |
| `harbor` (recommended) | One container per trial, a separate verifier, network policy, native skill install | Numbers you publish or compare |
| `native` | Host workspace, no isolation | Fast iteration on your own machine |

Harbor is the recommended backend because host isolation is not optional for a
skill benchmark. A public proof run on 2026-11-14 found five failures in the
native runner:

1. A no-skill agent walked up from its workspace and read the vendored
   `SKILL.md`, `evals.json`, and the grader.
2. An agent read sibling workspaces through its tool's shared snapshot store.
3. An agent fetched the skill from the internet and installed it globally. The
   install leaked into later runs.
4. Two arms shared `/tmp`.
5. Agent startup failures were graded as false zeros.

PR #16 fixed what the runner could fix. Isolation is Harbor's job: one container
per trial, tests uploaded only after the agent phase, a separate verifier by
default, and `no-network` during the agent run with an inference-host allowlist.

The native runner stays as the quick mode. It is faster to debug, but it is
**not isolated**. Do not use it for results you intend to publish.

## Version and documentation status

This README documents the source tree it lives in. If you installed
`agent-skill-eval` from PyPI or a different Git branch, check the installed CLI
before copying a command:

```bash
agent-skill-eval --version
agent-skill-eval --help
agent-skill-eval run-all --help
```

During active development, the newest CLI flags and workflow docs may exist only
on a local feature branch until that branch is pushed and released. To use a
local checkout explicitly, run through that checkout instead of whatever command
is first on `PATH`:

```bash
uv run --project /path/to/skill-evals agent-skill-eval run-all --help
uv run --project /path/to/skill-evals python -m agent_skill_eval validate-all --root .
```

When sharing results, record the command, `agent-skill-eval --version`, agent
model, reasoning effort, and branch/commit of the harness repo. The harness also
writes most of that to `run_meta.json` and `summary.json`, but copying it into a
handoff or PR description makes stale-doc mismatches much easier to spot.

## Quick Start

```bash
# 1. See what's available
agent-skill-eval list

# 2. Validate an eval suite
agent-skill-eval validate examples/write-release-notes/evals/evals.json

# 3. Run it (pin cheap models while iterating)
agent-skill-eval run \
  --skill ./skills/write-release-notes \
  --evals ./examples/write-release-notes/evals/evals.json \
  --agent claude-code --agent-model claude-code=haiku \
  --agent codex --agent-model codex=gpt-5-mini \
  --runs 3

# 4. Read the results
agent-skill-eval report --workspace ./eval-workspace/write-release-notes-workspace --show-evidence
```

For isolated, publishable numbers, run the same suite on Harbor:

```bash
agent-skill-eval run --backend harbor \
  --skill ./skills/write-release-notes \
  --evals ./examples/write-release-notes/evals/evals.json \
  --agent opencode \
  --agent-model opencode=openrouter/deepseek/deepseek-v4.1-flash \
  --runs 3 --concurrency 2
```

Harbor model strings need the provider prefix (`openrouter/...`) so Harbor can
route the key and endpoint. The task container installs the agent CLI; the host
only needs Docker and `OPENROUTER_API_KEY`.

To start a suite for your own skill:

```bash
agent-skill-eval init my-skill --template basic
```

This scaffolds `my-skill/SKILL.md` (frontmatter template), `my-skill/evals/evals.json`, and `my-skill/evals/files/` for fixtures. Templates are `basic`, `review-diff`, `commit-pr`, and `config-validator`; each includes a negative control and mostly deterministic assertions.

## Mental model

agent-skill-eval is a harness for measuring whether a skill changes real agent
behavior. It is not a raw model benchmark and it is not a static linter for
`SKILL.md`; it runs the same CLI a user would run, in a prepared workspace, and
then compares what changed.

The core objects are:

| Term | Meaning |
| --- | --- |
| Skill | A directory containing `SKILL.md` and optional `references/`, `scripts/`, or other support files. This is the thing being evaluated. |
| Eval suite | One `evals.json` file with `skill_name` and a list of cases for one skill. In plugin repos it usually lives at `plugins/<plugin>/skills/<skill>/evals/evals.json`. |
| Eval case | One prompt plus fixtures, assertions, trigger settings, and side-effect level. A case should test one behavior or failure mode. |
| Fixture | A file or directory copied into the agent workspace before the prompt runs. Fixtures make the workspace deterministic. |
| Workspace | The temporary git checkout the agent sees. It is either a fresh git repo or a clone of `--source-repo`. Every case gets its own workspace so state does not leak between cases. |
| Agent harness | The adapter that invokes `codex`, `claude`, `opencode`, or `fake` and normalizes output, logs, timing, tokens, and exit status. |
| Configuration | A named run variant such as `with_skill` or `without_skill` for one agent/model. With `--runs N`, each configuration has repeated `run-1`, `run-2`, ... attempts. |
| Assertion | A deterministic or LLM-graded expectation. Deterministic assertions inspect git state, files, logs, or JSON; unresolved prose goes to the LLM rubric. |
| Grading result | The `grading.json` verdict for one run: passed, failed, skipped, evidence, and grading method per assertion. |
| Side-effect level | A case label: `static`, `local-only`, `live-safe`, or `live-external`. It lets you run cheap or safe cases before any live provider workflow. |
| Lifecycle hook | A shell command run before the suite, after each run is graded, or after the suite. Hooks are how provider-specific setup, external checks, and teardown are attached. |
| Cleanup manifest | `cleanup.json`, a scoped list of branches and PRs/MRs created by the run. Cleanup only acts on this manifest. |
| Summary/report | `summary.json`, `benchmark.json`, `status`, `report`, `report-all`, and optional JUnit/HTML/Markdown output. These are the artifacts you use for decisions. |

One normal `run` flows like this:

1. Load the skill and eval suite; apply filters such as `--eval-id` and
   `--side-effect-level`.
2. Run `--pre-run-command` hooks. These usually validate the skill text, create
   or reset a scratch source repo, or fail fast before any agent call.
3. For each case, create a workspace, clone `--source-repo` or initialize a
   fresh repo, copy fixtures, and install the skill into the agent-specific
   skill path.
4. Capture pre-run git/provider state.
5. Invoke the real agent CLI with the case prompt. If baseline mode is enabled,
   run both `with_skill` and `without_skill`.
6. Capture post-run state, logs, timing, token telemetry, and output.
7. Grade deterministic assertions first, then send unresolved assertions to the
   LLM grader when credentials are available.
8. Run `--post-grade-command` hooks while the workspace still exists. These can
   append structured assertion results for provider state that the core harness
   cannot know about.
9. Delete or keep workspaces according to debug settings, write summaries, and
   run `--post-run-command` hooks for teardown and cleanup verification.

Use `force_skill_invocation: true` when debugging whether the skill instructions
are correct after the skill has already triggered. Use
`force_skill_invocation: false` for discovery cases that prove the agent chooses
the skill from a natural request. Use `should_trigger: false` for negative
controls that prove the skill does not activate on adjacent read-only prompts.

## Defining evals

```json
{
  "skill_name": "write-release-notes",
  "evals": [
    {
      "id": "explicit-invoke",
      "prompt": "Write release notes for the commit history in commits.txt.",
      "expected_output": "A RELEASE_NOTES.md grouping changes by type with breaking changes highlighted.",
      "files": ["files/commits.txt"],
      "force_skill_invocation": true,
      "assertions": [
        "The file `RELEASE_NOTES.md` exists",
        "The breaking change is mentioned prominently",
        "The release notes do not mention any change that is not in commits.txt"
      ]
    },
    {
      "id": "negative-control",
      "prompt": "How many commits are listed in commits.txt?",
      "expected_output": "The skill should NOT trigger.",
      "files": ["files/commits.txt"],
      "should_trigger": false,
      "assertions": [
        "A new git branch was created",
        "A git commit was created",
        "The agent only answered the question without creating files"
      ]
    }
  ]
}
```

A JSON Schema for this format ships at [`schemas/evals.schema.json`](schemas/evals.schema.json) — point your editor at it for autocompletion, and run `agent-skill-eval validate <file>` in CI.

### Eval fields

| Field | Type | Purpose |
| --- | --- | --- |
| `id` | int \| str | Unique id within the suite |
| `prompt` | str | Prompt sent to the agent verbatim |
| `expected_output` | str | Reference output for LLM rubric grading |
| `files` | list[str] | Fixture file paths (resolved relative to the evals directory) |
| `stage_files` | bool | Deprecated compatibility flag: fixtures are always committed into the workspace baseline before the agent runs, so this no longer changes behavior |
| `assertions` | list[str \| object] | Prose assertions or structured deterministic assertions to grade against |
| `should_trigger` | bool | If false, branch/commit/push/PR assertions are inverted (default: true) |
| `force_skill_invocation` | bool | If true, the prompt is prefixed with `Use the $<skill> skill.` (default: false) |
| `side_effect_level` | str | One of `static`, `local-only`, `live-safe`, `live-external` (default: `local-only`) |

Fixture paths are copied into the eval workspace before each run. Direct fixtures under `files/` are staged at the workspace root (`files/config.json` -> `config.json`). Case-scoped fixture folders also become the workspace root when a case uses one prefix (`files/prod-only/bootstrap/...` -> `bootstrap/...`), which lets suites model multi-repo workspaces without long prompt paths. If one eval references multiple case prefixes, those prefixes are preserved to avoid same-name collisions. Unsafe paths and target collisions fail during `validate`.

## How grading works

Each assertion is graded by the first matching method:

1. **Deterministic** — code-based checks against the pre/post git state delta and run logs. No LLM involved.
2. **LLM rubric** — anything the deterministic grader doesn't recognize goes to an LLM judge with the agent output, expected output, and workspace file listing.
3. **Skipped** — if no LLM grader is configured (missing API key) or the grader errors, the assertion is marked *skipped* and excluded from the pass rate, never silently failed.

### Recognized deterministic assertion patterns

The deterministic grader matches assertion text against these patterns (case-insensitive):

| Pattern in assertion text | Check performed |
| --- | --- |
| `branch` + `created`/`exists`/`new` | A new branch appeared in the state diff and is checked out |
| `commit` + `created`/`exists`/`new` | A new commit appeared and is on the current branch |
| `push` + `remote`/`branch`/`pushed` | The eval-created branch was pushed AND remote HEAD matches local HEAD |
| `pr` or `pull request` | A new open PR/MR targets the eval-created branch (corroborated by `gh pr view` for GitHub when available) |
| `file exists` or `created` + a filename | File matching the backticked/quoted name exists in the workspace |
| `ran` + a command name (`npm`, `git`, ...) | Command name appears in the run logs |
| `contains` or `includes` + `"quoted"`/`` `backticked` `` text | Agent output contains the text |
| `valid json` | Agent output (or a workspace file) parses as JSON |

Anything else falls through to the LLM rubric. With `should_trigger: false`, the branch/commit/push/PR/MR checks invert: they pass only when those artifacts did **not** appear.

### Structured deterministic assertions

For checks that should never fall through to the LLM rubric, use assertion objects:

```json
{
  "type": "file_contains",
  "path": "README.md",
  "contains": "Quickstart",
  "text": "README includes a Quickstart section"
}
```

Supported `type` values are `file_exists`, `file_not_exists`, `file_contains`, `file_not_contains`, `file_unchanged`, `command_ran`, `command_not_ran`, `tests_pass`, `git_clean`, `json_path`, and `json_path_not_exists`. Common fields are `path`, `contains`, `command`, `json_path`, `equals`, and optional human-facing `text`.

## CLI Commands

### `export-harbor`

```bash
agent-skill-eval export-harbor \
  --skill <skill-dir> --evals <evals.json> --out <dir> \
  [--base-image python:3.12-slim] [--workdir /app] [--force]
```

Writes one Harbor task per eval case. Each task has `task.toml`, `instruction.md`, an `environment/` with committed fixtures, and a `tests/` directory with the deterministic grader and a baseline manifest. Forced-invocation cases get a `__with` and a `__without` variant, so the with-skill prompt prefix never leaks into the baseline.

`run --backend harbor` calls the exporter for you. Use `export-harbor` directly when you want to inspect or lint the tasks first:

```bash
for task in <dir>/tasks/*/; do harbor lint "$task"; done
```

### `run`

```bash
agent-skill-eval run \
  --skill <skill-dir> --evals <evals.json> \
  --agent opencode --agent claude-code --agent codex \
  [--backend native|harbor] \
  [--agent-model claude-code=haiku] [--agent-model codex=gpt-5-mini] \
  [--reasoning-effort medium] \
  [--harness-base-url https://openrouter.ai/api/v1] \
  [--runs 3] [--concurrency 2] [--iteration 1] \
  [--baseline/--no-baseline] [--cleanup] \
  [--timeout 600] [--retries 1] \
  [--eval-id wif-too-long-stop] \
  [--side-effect-level static] \
  [--max-total-tokens-per-case 200000] [--budget-action fail] \
  [--pre-run-command "python scripts/setup.py"] \
  [--post-grade-command "python scripts/grade_external.py"] \
  [--post-run-command "python scripts/teardown.py"] \
  [--pricing-config pricing.json] \
  [--grader-model deepseek/deepseek-v4.1-flash] [--grader-base-url URL] \
  [--harbor-base-image python:3.12-slim] [--harbor-jobs-dir DIR] \
  [--write-baseline baseline.json] [--compare baseline.json] \
  [--source-repo https://github.com/foo/bar.git] \
  [--workspace ./eval-workspace]
```

Key options:

- `--backend`: `native` (host workspace, not isolated) or `harbor` (Docker, isolated). Harbor ignores `--source-repo`, budget checks, pricing, and lifecycle hooks; it prints a warning for each.
- `--agent-model, -m`: model per agent as `agent=model`; a bare value applies to all agents. Repeatable. With Harbor, use the provider-prefixed form (`openrouter/deepseek/deepseek-v4.1-flash`).
- `--write-baseline` / `--compare`: write a baseline JSON after the run, or compare the run to an earlier baseline. The comparison includes per-config pass-rate, pass^k, token, cost, and time deltas, plus a per-case bootstrap interval, and is written to `comparison.json` and `comparison.md`.
- `--reasoning-effort`: reasoning setting passed to agents that support it (codex: `model_reasoning_effort`, e.g. `minimal`/`low`/`medium`/`high`/`xhigh`). Without it, agents silently inherit their own local config (e.g. `~/.codex/config.toml`) — two machines can then benchmark different reasoning settings under the same command line. Agents without a pass-through flag print a warning and record `null`. The effective value lands in `run_meta.json`.
- `--harness-base-url`: injected as `ANTHROPIC_BASE_URL` (claude-code) / `OPENAI_BASE_URL` (codex, opencode) into the agent process.
- `--runs, -n`: repeat each (eval, agent, config) N times; enables pass@k and pass^k stats with a Wilson interval. Results land in `run-1/`, `run-2/`, ... subdirectories.
- `--timeout` / `--retries`: per-run agent timeout (default 600s) and retries on timeout or non-zero exit (default 1). Also settable via `ASE_AGENT_TIMEOUT` / `ASE_AGENT_RETRIES`.
- `--eval-id`: run only the named case(s); repeatable. Unknown ids fail fast with the list of available ids, before any model call. The applied filter is recorded in `evals_meta.json` (`selected_eval_ids`) and `summary.json` (`eval_ids`), and all artifacts naturally describe only the selected cases. The cheap inner loop after a small skill edit is `validate --eval-id X` then `run --eval-id X`.
- `--side-effect-level`: run only the requested side-effect level(s); repeatable. Use this to run `static` and `local-only` checks before deliberate live-provider suites.
- Budget guards (`--max-input-tokens-per-case`, `--max-non-cached-input-tokens-per-case`, `--max-output-tokens-per-case`, `--max-total-tokens-per-case`, `--max-reasoning-tokens-per-case`, `--max-cost-per-case`, `--max-duration-per-case`, `--budget-action warn|fail|stop-suite`): per-case limits checked after each agent run (mid-run kills remain `--timeout`'s job — budgets exist because a case can finish within the timeout and still be far too expensive). A violation is recorded in `timing.json` (`budget_exceeded`, `budget_reason`); with `fail` (the default) it also appends a failed `method: "budget"` assertion to the run's `grading.json`, and with `stop-suite` it additionally skips every run that hasn't started yet — useful to stop a live side-effect suite before more remote branches/PRs get created. `--max-cost-per-case` only fires when the run has a cost source (see Cost reporting); likewise `--max-reasoning-tokens-per-case` and `--max-non-cached-input-tokens-per-case` only fire when the run reported those values — an agent exposing less telemetry is never failed over an unknown quantity.
- Lifecycle hooks (`--pre-run-command`, `--post-grade-command`, `--post-run-command`): see [Lifecycle hooks](#lifecycle-hooks).
- `--pricing-config`: JSON file with USD-per-token rates for agents whose CLI reports no cost (see Cost reporting).

While the suite runs, each case prints start/finish lines (elapsed, tokens in/cached/out, cost), and `progress.jsonl` in the iteration directory gets one JSON event per line (`suite_started`, `run_started`, `agent_started`, `agent_finished`, `run_finished`, `budget_exceeded`, ...) — `tail -f` it from another shell to watch a long run.

### `report`

```bash
agent-skill-eval report --workspace <skill-workspace> [--iteration N] [--format table|markdown|html] [--show-evidence] [--failures-only]
```

`--show-evidence` prints the evidence string for every failed or skipped assertion — the state diff for deterministic checks, the judge's reasoning for LLM checks. `--failures-only` limits the per-eval detail to runs that had at least one failed assertion. `--format markdown` emits a paste-ready table; `--format html` emits a standalone static HTML report.

### `run-all`

```bash
agent-skill-eval run-all \
  --root path/to/plugin-repo --agent codex --side-effect-level static \
  [--source-repo https://github.com/foo/bar.git] \
  [--max-total-tokens-per-case 200000] [--budget-action fail] \
  [--pre-run-command "python scripts/setup.py"] \
  [--post-grade-command "python scripts/grade_external.py"] \
  [--post-run-command "python scripts/teardown.py"] \
  [--pricing-config pricing.json] \
  [--suite-concurrency 4] \
  [--plan-only] \
  [--changed-only --base-ref origin/main] \
  [--junit-xml run-all.xml] [--fail-on-assertions] [--min-pass-rate 0.8]
```

Runs every paired `SKILL.md` + `evals/evals.json` suite discovered under a repo by invoking the normal single-suite runner once per skill. Suites with no cases for the requested side-effect filter are recorded as skipped in `run-all-iteration-N.json`, which makes `static` or `local-only` repo sweeps safe before any live-provider evals.

Repo sweeps accept the same source-repo, model, timeout/retry, budget, pricing, and lifecycle hook controls as `run`; the aggregate manifest records the source repo, budget, pricing config path, suite concurrency, and hook counts used for the sweep. Use `--plan-only` to validate and write the planned/skipped suite manifest without running agents, `--suite-concurrency` to run independent skill suites in parallel while preserving deterministic manifest ordering, `--changed-only --base-ref origin/main` in CI to run only skills or eval suites touched by the branch, `--junit-xml` to publish the sweep as test results, and `--fail-on-assertions` or `--min-pass-rate` to make quality gates explicit.

### `report-all`

```bash
agent-skill-eval report-all --workspace ./eval-workspace [--format table|markdown|html|json] [--failures-only]
```

Summarizes a repo-level `run-all` manifest with suite status, pass rates, duration, token totals, cost totals, and failure details. Use `--format markdown` for MR notes, `--format html` for a static artifact, and `--format json` when another tool should consume the raw manifest.

### `rerun-failed`

```bash
agent-skill-eval rerun-failed --from ./eval-workspace/run-all-iteration-1.json
```

Prints targeted `agent-skill-eval run --eval-id ...` commands for failed repo-sweep cases. The generated commands preserve run context recorded in the manifest, including grader settings, agent model, source repo, pricing config, timeout/retry, reasoning effort, and budget flags.

### `status`

```bash
agent-skill-eval status --workspace <skill-workspace> [--iteration N]
```

Answers "what happened in the last run?" without a model call: pass rates per configuration, failed/skipped/errored runs, token totals with the cached share, total cost (or `unavailable`), budget violations, hook outcomes, and whether `cleanup.json` recorded remote side effects. Reads the iteration's `summary.json`; for runs that predate it, a reduced summary is rebuilt from the saved artifacts.

`summary.json` also reports `trigger_rate` per configuration: whether the skill was actually loaded, based on OpenCode `tool_use` events (the `skill` tool or a read of the installed `SKILL.md`) and Codex command events. Runs where the harness exposes too little telemetry are counted as `unknown` and are excluded from the rate denominator — unknown is never treated as "did not trigger". Each run records `skill_invoked` and `skill_invocation_evidence` in `run_meta.json`.

### `compare`

```bash
agent-skill-eval compare --workspace <skill-workspace> 1 2
```

Side-by-side pass rates of two iterations, per configuration, with pass-rate, time, and token deltas — the feedback loop for iterating on a SKILL.md.

### `validate`

```bash
agent-skill-eval validate path/to/evals.json [--eval-id <id>]
```

Schema check plus referenced-fixture existence, duplicate-id detection, and structured-assertion required-field/path checks. Exit code 1 on any problem (CI-friendly). `--eval-id` validates only the selected case(s) (duplicate-id detection still covers the whole file); pair it with `run --eval-id` to confirm a targeted case is well-formed before spending a model call on it.

### `validate-all`

```bash
agent-skill-eval validate-all --root path/to/repo [--format table|json]
```

Recursively validates every discovered `*/evals/evals.json` under a repo and exits non-zero if any suite is invalid. This is the CI-friendly check for plugin repos with many skills.
For plugin-style paths such as `plugins/<plugin>/skills/<skill>/evals/evals.json`, it also verifies the adjacent `SKILL.md` frontmatter and that the suite `skill_name` matches the skill directory.

### `doctor`

```bash
agent-skill-eval doctor --skill ./skills/my-skill --evals ./evals/evals.json --agent codex
```

Preflights the local agent CLI, skill frontmatter, eval suite, LLM-grader credentials, and source-repo tooling (`gh` for GitHub, `glab` for GitLab). Use it before live or expensive suites to catch missing tools and auth early. Missing `name`/`description` frontmatter fields are failures; a frontmatter `name` that differs from the skill directory name is only a warning.

### `list`

```bash
agent-skill-eval list [--root .] [--format table|json]
```

Recursively discovers eval suites (`*/evals/evals.json`) and skills (`SKILL.md`) under a directory, including plugin-style layouts such as `plugins/<plugin>/skills/<skill>/evals/evals.json`.
JSON output includes a `pairs` section showing paired skills, skills without evals, and external eval suites.

### `grade`

```bash
agent-skill-eval grade --workspace <iteration-dir> [--recompute-benchmark]
```

Re-grades existing outputs using saved `evals_meta.json` and state snapshots. Two caveats: LLM-graded assertions are re-evaluated from scratch and may flip verdicts, and because the original agent workspace is deleted after the run, the judge re-grades from the saved artifacts (agent output, logs) rather than the live workspace files.

### `cleanup`

```bash
agent-skill-eval cleanup --workspace ./eval-workspace [--yes]
```

Only closes PRs/MRs and deletes remote branches recorded in `cleanup.json`. Never touches unrelated review requests, branches, or workspaces.

### `init`

```bash
agent-skill-eval init my-skill [--output ./examples] [--template basic|review-diff|commit-pr|config-validator]
agent-skill-eval init my-skill --layout plugin --plugin demo-plugin --output /path/to/plugin-repo
```

Use `commit-pr` for live branch/push/PR workflows, `review-diff` for read-only review skills, and `config-validator` for skills that inspect bundled config fixtures.
Use `--layout plugin` to scaffold directly into `plugins/<plugin>/skills/<skill>/evals` for Codex plugin repositories.

## Lifecycle hooks

Skills with external side effects (push, PR/MR creation, provider APIs) need setup, provider-specific grading, and teardown that the harness can't know about. Instead of hard-coding providers, `run` exposes three shell-command hook points; each receives run metadata as `ASE_*` environment variables on top of the parent environment:

- `--pre-run-command` (repeatable): runs once before any agent case — seed or reset a scratch repo, run static checks on the skill text. A non-zero exit **aborts the suite before a single model call**. Env: `ASE_SKILL_NAME`, `ASE_SKILL_DIR`, `ASE_EVALS_PATH`, `ASE_ITERATION`, `ASE_ITERATION_DIR`, `ASE_WORKSPACE_BASE`, `ASE_SOURCE_REPO`, `ASE_RUN_ID`, `ASE_EVAL_IDS` (comma-separated).
- `--post-grade-command` (repeatable): runs once per (eval, agent, config) run, after `grading.json` is written and **before the workspace is deleted** — verify remote state, check side-effect files. On top of the suite-level vars it gets `ASE_EVAL_ID`, `ASE_AGENT`, `ASE_WITH_SKILL` (`1`/`0`), `ASE_RUN_INDEX`, `ASE_WORKSPACE_PATH`, `ASE_OUTPUT_DIR`, `ASE_PRE_STATE_PATH`, `ASE_POST_STATE_PATH`, `ASE_TIMING_PATH`, `ASE_RUN_META_PATH`, `ASE_GRADING_PATH`. If the hook prints a JSON array of `{"text": ..., "passed": ..., "evidence": ...}` objects to stdout, they are appended to the run's `grading.json` as `method: "hook"` assertion results and the summary is recomputed — external checks land in the same artifact `report` reads. A non-zero exit appends one failed hook check. Raw hook output is saved to `outputs/post_grade_hooks.json`.
- `--post-run-command` (repeatable): runs once after the suite (teardown, cleanup verification). Failures are recorded in `summary.json` and printed, but don't fail the run — the results already exist.

Hook commands run through the shell from the invocation directory, with a timeout of `ASE_HOOK_TIMEOUT` seconds (default 600).

```bash
agent-skill-eval run --skill ./skills/my-skill --evals ./evals/evals.json --agent codex \
  --eval-id feature-branch \
  --pre-run-command  "python scripts/setup_source_repo.py" \
  --post-grade-command "python scripts/grade_remote_state.py" \
  --post-run-command "python scripts/teardown_remote.py"
```

## Live side-effect cookbook

Use this pattern for skills that create external state: pushes, PRs/MRs, Jira
issues, cloud resources, provider API changes, tickets, comments, or anything
that cannot be fully judged from the local workspace.

### 1. Classify cases by blast radius

Set `side_effect_level` on every case:

| Level | Use for |
| --- | --- |
| `static` | Prompt-shaping, validation, "ask for missing input", and no-tool checks. No local git writes or provider calls. |
| `local-only` | Local file edits, local git branches/commits, dry-run provider payloads, or live calls only to isolated scratch projects that are fully cleaned by hooks. |
| `live-safe` | Live provider state in a controlled test project where setup and teardown are reliable and the effect is reversible. |
| `live-external` | Shared, customer, production, paid, or human-visible systems. Run only with explicit intent and tight filters. |

Start with `static` and `local-only`. Add `live-safe` only after the local cases
prove the workflow shape. Treat `live-external` as a release candidate check,
not an inner loop.

### 2. Make the source repo disposable

For git/MR workflows, use `--source-repo` pointing at a scratch repository that
the eval owns. The pre-run hook should reset it to a known base state before
each suite or case:

```bash
agent-skill-eval run \
  --skill ./plugins/acme/skills/commit-push-mr \
  --evals ./plugins/acme/skills/commit-push-mr/evals/evals.json \
  --source-repo git@gitlab.example.com:me/agent-skill-eval-scratch.git \
  --agent codex \
  --side-effect-level local-only \
  --eval-id wif-too-long-stop \
  --pre-run-command "python plugins/acme/skills/commit-push-mr/evals/scripts/setup_source_repo.py" \
  --post-grade-command "python plugins/acme/skills/commit-push-mr/evals/scripts/grade_gitlab.py" \
  --post-run-command "python plugins/acme/skills/commit-push-mr/evals/scripts/teardown_gitlab.py" \
  --post-run-command "python plugins/acme/skills/commit-push-mr/evals/scripts/verify_gitlab_clean.py"
```

The scratch repo should contain only eval fixtures and expected provider state.
Do not point live evals at the real project until the suite has already passed
against scratch state.

### 3. Keep setup, grading, teardown separate

Use one script for each job:

- Setup hook: creates or resets scratch state, checks auth, validates that the
  target is the expected test project, and aborts before model calls if anything
  is unsafe.
- Post-grade hook: reads `ASE_*` paths, inspects provider state while the agent
  workspace still exists, and prints JSON assertion rows to stdout.
- Teardown hook: closes/deletes only expected branches, MRs, tickets, labels, or
  resources created by this suite.
- Verify-clean hook: read-only proof that no expected eval branch, MR, ticket, or
  resource remains. Run it after teardown and independently after failures.

Post-grade hooks can append checks to `grading.json` by printing:

```json
[
  {
    "text": "Exactly one GitLab MR was opened for the eval branch",
    "passed": true,
    "evidence": "MR !42 targets main and source branch proj-123-eval"
  }
]
```

This keeps external verification in the same report as deterministic file/git
checks and LLM rubric checks.

### 4. Write cases around failure modes, not just happy paths

A good side-effect suite usually has these case types:

- Happy path: creates the intended branch/change/review request/resource in the
  scratch provider.
- Guardrail stop: invalid branch name, missing ticket ID, missing required
  input, unsupported environment, or too-large payload stops before side
  effects.
- No-op stop: no meaningful local changes means no commit, push, ticket, or API
  write.
- Natural trigger: same workflow without `force_skill_invocation`, proving the
  agent selects the skill from a realistic user request.
- Negative control: adjacent read-only prompt with `should_trigger: false`,
  proving no side effects are created.
- Cleanup regression: provider state is clean after teardown, even when the
  agent failed earlier.

For each side effect, pair a positive assertion with an explicit "nothing else"
check in the provider hook. For example: one MR for the eval branch, no MR for
`main`, no duplicate MR, no unexpected branch, and no untracked provider
resource.

### 5. Use a cheap inner loop

Do not make full live runs your edit loop. Use this order:

```bash
agent-skill-eval validate path/to/evals.json --eval-id wif-too-long-stop
agent-skill-eval doctor --skill path/to/skill --evals path/to/evals.json --agent codex
agent-skill-eval run --skill path/to/skill --evals path/to/evals.json \
  --agent codex --eval-id wif-too-long-stop --side-effect-level local-only \
  --no-baseline --max-total-tokens-per-case 150000 --budget-action fail
agent-skill-eval status --workspace ./eval-workspace/<skill>-workspace
agent-skill-eval report --workspace ./eval-workspace/<skill>-workspace --show-evidence
```

Only after a targeted case passes should you run more cases, more agents,
baseline mode, or `run-all`.

For plugin repos with many skills, use planning before execution:

```bash
agent-skill-eval validate-all --root /path/to/plugin-repo --strict
agent-skill-eval run-all --root /path/to/plugin-repo \
  --side-effect-level static --changed-only --base-ref origin/main --plan-only
agent-skill-eval run-all --root /path/to/plugin-repo \
  --side-effect-level local-only --changed-only --base-ref origin/main \
  --suite-concurrency 2 --fail-on-assertions
```

### 6. Use budgets and cleanup as gates

Side-effect suites need explicit failure gates:

- Set `--timeout`, `--retries`, `--max-total-tokens-per-case`,
  `--max-duration-per-case`, and usually `--budget-action fail` or
  `stop-suite`.
- Use `--eval-id` while debugging a single risky case.
- Keep `--cleanup` or teardown hooks enabled for any run that can push, open an
  MR, or write provider state.
- Run a read-only verify-clean command after failures before starting another
  live run.
- Archive or paste `status`, `report --show-evidence`, and provider hook
  evidence when claiming a skill improved.

The minimum credible claim for a side-effect skill is: targeted case passed,
provider hook passed, cleanup verification passed, and the command/model/harness
version are recorded. A stronger claim adds repeated runs, natural trigger and
negative-control cases, and a before/after comparison with `compare`.

## Example skills

Five example skills ship with the repo, chosen to exercise different grading surfaces:

| Skill | Tests | Grading surface |
| --- | --- | --- |
| [`commit-push-pr`](skills/commit-push-pr/SKILL.md) | git workflow automation | deterministic state-diff checks (branch/commit/push/PR); needs a `--source-repo` |
| [`fix-failing-tests`](skills/fix-failing-tests/SKILL.md) | error recovery / iterative refinement | deterministic + file checks; fully offline |
| [`write-release-notes`](skills/write-release-notes/SKILL.md) | subjective writing quality, anti-fabrication | LLM rubric grading; fully offline |
| [`validate-config`](skills/validate-config/SKILL.md) | bundled resources: does the agent run the skill's `scripts/` and read its `references/`? | command-ran + file + content checks; fully offline |
| [`review-diff`](skills/review-diff/SKILL.md) | read-only analysis: planted bugs found, documented decoy not flagged, nothing modified | chat-output-only grading (content + LLM rubric); fully offline |

Each has a matching eval suite under [`examples/`](examples/). `skills/` holds the artifacts being evaluated; `examples/` holds the test cases — so you can test one skill against many suites or one suite against many skill versions.

## Reading results

### Workspace layout

```
eval-workspace/
└── <skill>-workspace/
    └── iteration-1/
        ├── evals_meta.json          # eval definitions, selected filters, skill_hash (used by `grade`)
        ├── cleanup.json             # manifest of artifacts created by this run
        ├── benchmark.json           # aggregate stats per (agent, config)
        ├── summary.json             # one-file run summary (used by `status`)
        ├── progress.jsonl           # live per-run events; tail -f during the run
        └── eval-<id>/<agent>/<config>/   # config = with_skill | without_skill
            ├── run-N/               # only when --runs > 1
            ├── outputs/
            │   ├── output.txt       # final agent output
            │   ├── stdout.log / stderr.log
            │   ├── post_grade_hooks.json   # only with --post-grade-command
            │   └── pre_state.json / post_state.json
            ├── timing.json          # tokens (incl. non-cached/reasoning splits), duration, exit_code, timed_out, retries, budget verdict
            ├── grading.json         # per-assertion results
            ├── trajectory.json      # normalized lifecycle events for this run
            └── run_meta.json        # agent, with_skill, run_index, model, side_effect_level, skill_hash, CLI/harness versions, ...
```

### grading.json

```json
{
  "assertion_results": [
    {
      "text": "A new git branch was created",
      "passed": false,
      "method": "deterministic",
      "skipped": false,
      "evidence": "No new branch appeared in this run. current_branch='main'"
    }
  ],
  "summary": {"passed": 2, "failed": 1, "skipped": 0, "total": 3, "pass_rate": 0.667}
}
```

To debug a failure: find `"passed": false` entries, read `evidence`, compare `pre_state.json`/`post_state.json`, then check `outputs/output.txt` for the agent's full response. Or just run `agent-skill-eval report --show-evidence`.

### Skill-centric metrics

`benchmark.json` and `summary.json` answer more than "did it pass":

- `pass_rate` is the mean per-run pass rate. `pass_hat_k` is the stricter
  per-case metric: **all k attempts pass**. It has a Wilson interval, the case
  count, and the number of skipped assertions. `pass_at_k` ("any attempt
  passes") is kept next to it.
- `stats.tokens`, `stats.cost_usd`, and `stats.time_seconds` carry `median` and
  `p90` next to `mean`, so a change that keeps correctness but doubles cost or
  time is visible.
- `summary.tag_pass_rates` reports pass rates per case tag: `correctness`,
  `skill_effect`, or `judge`. Untagged cases default to `correctness`.
- `summary.trigger_rate` splits positive cases from negative controls
  (`should_trigger: false`). `unknown` is reported separately and never counted
  as "did not trigger".
- `summary.out_of_workspace` counts runs whose ATIF tool calls touched paths
  outside the task workdir, with examples. It is a leak signal inside the
  sandbox, not a hard failure.
- Runs with 0 output tokens, an invalid Harbor reward, or an agent-side
  exception are errors, never scores. They appear in `summary.errors`.

### Iterating on a skill

1. `agent-skill-eval run ...` → 2. `agent-skill-eval report --show-evidence` → 3. edit SKILL.md → 4. `agent-skill-eval run --iteration 2 ...` → 5. `agent-skill-eval compare --workspace ... 1 2`

### Comparing agents: reading with/without-skill deltas

Run several agents in one invocation and every agent gets its own baseline comparison:

```bash
agent-skill-eval run \
  --skill ./skills/fix-failing-tests \
  --evals ./examples/fix-failing-tests/evals/evals.json \
  --agent claude-code --agent-model claude-code=claude-haiku-4-5-20251001 \
  --agent opencode --agent-model opencode=deepseek/deepseek-v4-flash:free \
  --runs 3

agent-skill-eval report --workspace ./eval-workspace/fix-failing-tests-workspace --format markdown
```

The report shows one row per (agent, config) and a delta per agent (numbers below are illustrative):

| Configuration | Pass Rate | Full Pass / pass@k | Time (s) | Tokens | Cost (USD) |
| --- | --- | --- | --- | --- | --- |
| claude-code_with_skill | 91.7% +/- 14.4% | 67% (k=3) | 41.2 +/- 8.0 | 1840 +/- 312 | 0.0042 +/- 0.0011 |
| claude-code_without_skill | 58.3% +/- 14.4% | 33% (k=3) | 52.7 +/- 12.1 | 2410 +/- 405 | 0.0058 +/- 0.0019 |
| opencode_with_skill | 83.3% +/- 0.0% | 67% (k=3) | 64.9 +/- 9.3 | 2980 +/- 220 | 0.0000 +/- 0.0000 |
| opencode_without_skill | 50.0% +/- 25.0% | 33% (k=3) | 71.5 +/- 15.8 | 3340 +/- 510 | 0.0000 +/- 0.0000 |

**Delta (with_skill - without_skill):**

- `claude-code`: pass rate +33.3%, time -11.5s, tokens -570, cost -0.0016 USD
- `opencode`: pass rate +33.3%, time -6.6s, tokens -360, cost +0.0000 USD

How to read it: the **delta rows** are the skill's measured value per agent — here the skill lifts pass rate by ~33 points on both agents *and* saves time/tokens, the strongest possible signal. A positive pass-rate delta with a large token increase means the skill works but is verbose; a near-zero delta means that agent doesn't benefit (check `report --show-evidence` to see whether it never triggered the skill). The same numbers are machine-readable in `benchmark.json`: per-config stats under `run_summary`, per-agent deltas under `deltas`, keyed by agent name.

## Environment variables

- `OPENROUTER_API_KEY` / `OPENAI_API_KEY`: API key for LLM rubric grading
- `OPENAI_BASE_URL`: custom grader endpoint (defaults to OpenRouter)
- `ASE_AGENT_TIMEOUT` / `ASE_AGENT_RETRIES`: harness timeout/retry defaults
- `ASE_HOOK_TIMEOUT`: lifecycle hook command timeout in seconds (default 600)
- `ASE_KEEP_WORKSPACE`: keep per-eval workspaces for debugging

## Agent-specific details

| Agent | Command | Skill install path |
| --- | --- | --- |
| OpenCode | `opencode run --format json --dangerously-skip-permissions` | `.opencode/skills/<name>/SKILL.md` |
| Claude Code | `claude -p --output-format json --dangerously-skip-permissions` | `.claude/skills/<name>/SKILL.md` |
| Codex | `codex exec --json --sandbox workspace-write --skip-git-repo-check` | `.agents/skills/<name>/SKILL.md` (also mirrored to `.codex/skills` for older builds) |

### Cost reporting

`timing.json` records `cost_usd` per run, taken from the agent CLI itself (claude: `total_cost_usd`, opencode: per-step `cost`). When no cost is available, `cost_usd` is `null` — **unavailable, not $0.00** — and reports/`benchmark.json` show `n/a` instead of a misleading zero. Cost stats are computed only over the runs that reported a cost (`cost_runs` in `benchmark.json`, `runs_with_cost` in `summary.json`).

Codex's CLI reports token counts but no cost. To get cost numbers anyway, pass `--pricing-config pricing.json` with your provider's USD-per-token rates, keyed by the exact `--agent-model` value (fields mirror OpenRouter's pricing schema):

```json
{
  "gpt-5.4-mini": {
    "prompt": 2.5e-07,
    "completion": 2e-06,
    "input_cache_read": 2.5e-08
  }
}
```

The computed value lands in `cost_usd` with `cost_usd_source: "pricing-config"`. Cached input is billed at `input_cache_read` (falling back to `prompt`); note that codex reports `input_tokens` *including* cache reads while claude-code and opencode report them separately — the harness accounts for this per agent. `summary.json` splits the totals (`tokens.cached`, `tokens.non_cached_input`, `tokens.cached_pct`), because on long agent sessions most input tokens are cache reads and the raw `input_tokens` number wildly overstates billable spend. Each run's `timing.json` also persists `non_cached_input_tokens` (computed with that run's CLI semantics) and `reasoning_output_tokens` when the agent reports reasoning telemetry (codex `turn.completed`, opencode `step_finish`) — `null` means unreported, not zero.

One sharp edge: the claude CLI prices runs at **Anthropic list prices** regardless of the endpoint it talks to. When you route claude-code through OpenRouter (`--harness-base-url` or `ANTHROPIC_BASE_URL`), agent-skill-eval reconciles the cost by recomputing it from the run's token counts and OpenRouter's published per-model rates. `timing.json` then shows:

- `cost_usd_source`: `"cli"` (the CLI's own number), `"openrouter-pricing"` (reconciled), `"pricing-config"` (computed from `--pricing-config`), or `"cli-unreconciled"` (reconciliation failed — e.g. a short model alias like `haiku` that can't be mapped to an OpenRouter slug — so `cost_usd` is the CLI's list-price estimate and actual billing differs)
- `cost_usd_cli`: the CLI's original estimate, kept alongside the reconciled value

Pin full model IDs (`claude-haiku-4-5-20251001`, not `haiku`) to keep reconciliation working.

## Development

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check src/ tests/
uv run --extra dev ruff format --check src/ tests/
```

A `fake` agent type exists for offline testing of the full run→grade→report pipeline (used by the CI smoke test). To record a demo cast: `./scripts/record-demo.sh`.

## License

MIT
