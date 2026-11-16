# ADR 0001: Harbor backend for skill-centric evaluation

- **Status:** accepted
- **Date:** 2026-11-15
- **Issue:** [#17](https://github.com/tardigrde/agent-skill-eval/issues/17)
- **Supersedes:** nothing

## Context

`agent-skill-eval` runs agent evaluations directly on the host. The runner
creates a workspace, installs a skill, runs an agent CLI, snapshots git state,
and grades the result. This design is fast and easy to debug. It is not
isolated.

A public proof run on 2026-11-14 found five failures:

1. A no-skill agent walked up from its workspace and read the vendored
   `SKILL.md`, `evals.json`, and the grader.
2. An agent read sibling workspaces through the agent tool's shared snapshot
   store.
3. An agent fetched a skill from the internet and installed it globally. The
   install leaked into later runs.
4. Two arms shared `/tmp`.
5. Agent startup failures were graded as false zeros.

PR #16 fixed the runner-level causes it could. Isolation is the remaining gap.
Harbor already provides the generic isolation: one container per trial, a
separate verifier, network policy, native skill install, ATIF trajectories, and
concurrency.

This ADR records the decisions for the port. It does not describe every Harbor
feature. The scope document is issue #17.

## Decision

Use Harbor as the isolation and execution backend. Keep `agent-skill-eval` as
the skill-centric evaluation layer:

- paired with-skill and without-skill arms,
- trigger rate,
- correctness and skill effect,
- pass^k,
- cost and time next to correctness.

The native runner stays as a quick mode. It is documented as not isolated.

## Pinned versions

| Component | Version | Source |
| --- | --- | --- |
| Harbor | 0.26.0 | PyPI `harbor==0.26.*` |
| Python | 3.12+ for the Harbor backend (3.12.9 used for the spike); the native runner keeps 3.11 | `pyproject.toml` |
| OpenCode V2 | 2.3.0 | Installed by Harbor's `opencode-v2` adapter |
| Agent model (spike) | `openrouter/deepseek/deepseek-v4.1-flash` | OpenRouter |
| Judge model (default) | `deepseek/deepseek-v4.1-flash` | OpenRouter |

Harbor uses a `0.x` release policy. A minor release may contain a breaking
change. Pin the minor version. Do not import Harbor internals. Use the CLI and
the job-directory contract.

## Concept mapping

| agent-skill-eval | Harbor |
| --- | --- |
| Eval case (`prompt`, `files`) | Task: `instruction.md` and `environment/` |
| Suite | A directory of tasks (implicit local dataset) |
| with-skill / without-skill arm | Two jobs with the same tasks; one job has `agents[].skills` |
| Structured assertions | `tests/test.sh` runs our deterministic grader and writes `reward.json` |
| `[JUDGE]` / LLM rubric | Host-side after import, using our `LLMGrader` |
| `--post-grade-command` | Verifier for workspace checks; importer for output checks |
| Git state-diff checks | Baseline tree plus workspace artifact; remote effects host-side |
| `--runs N` | `n_attempts` |
| `--concurrency` | `n_concurrent_trials` |
| Timing and token telemetry | `result.json` (`agent_result`, phase timings) |
| Trigger rate from stdout | ATIF tool calls plus observations |
| Cost | `agent_result.cost_usd` and ATIF `final_metrics.total_cost_usd` |

## Decisions

### Q1. Two jobs, not two agent entries

`TrialConfig.generate_trial_name()` builds `<task-name>__<7 random chars>`
(`src/harbor/models/trial/config.py`). Trial directories never collide. The arm
is recoverable from `result.json` -> `config.agent.skills`, and from `lock.json`
-> `skills[].digest`.

Harbor groups rewards by **agent name + model + dataset**
(`JobStats.format_agent_evals_key`). Two agent entries with the same adapter and
model land in one group. Harbor's own mean and viewer then mix the arms.

Decision: run `"<suite>__with_skill"` and `"<suite>__without_skill"` as two
jobs. Join them with our own run id. Environment builds are cached, so the extra
cost is orchestration.

`n_attempts` records no attempt index. Pass^k groups attempts by
`(task, arm)` count. Do not plan attempt-level pairing across arms.

### Q2. Trigger detection: ATIF first, per-agent parsers second

A trial counts as triggered only when the skill is loaded. These are not
triggers:

- the skill name or description in the system prompt,
- a skill file in the environment that the agent never opens,
- a skill tool call that failed.

Primary signal, from ATIF:

1. A tool call whose `function_name` is a skill tool (`skill`, `Skill`,
   `activate_skill`).
2. A tool call whose `arguments` contain `<skill>/SKILL.md` or the registered
   skills directory.
3. A read tool call whose arguments target a file under the skills directory.
4. An OpenCode `skill` call whose observation contains
   `<skill_content name="<skill-name>">`.

Rule 4 is necessary. Harbor's OpenCode V2 ATIF conversion records the skill
tool input as `{"id": "skill"}`, not the skill name. The name is only in the
observation content. This was confirmed in the M0 spike.

Fallback: the per-agent parsers from PR #16 (`detect_skill_invocation`). They
read raw CLI logs when ATIF is missing.

Unknown is not false. If ATIF is missing, the agent lacks skill support, or the
tool events are incomplete, the trigger state is `None`. The report shows the
unknown count. The trigger-rate denominator is the count of known runs.

`force_skill_invocation` cases are excluded from the natural trigger rate. They
test execution, not discovery.

### Q3. Git state: baseline tree, workspace artifact, host-side remote checks

A separate verifier gets a fresh copy of the agent environment and `tests/`.
Declared artifacts are restored to their original paths. `/logs/agent/` is not
copied.

Decision:

1. Commit fixtures in `environment/Dockerfile` with a fixed author and date.
   Tag the baseline commit.
2. Export a baseline manifest (`tests/baseline.json`) with the git tree hash,
   file hashes, and modes.
3. Declare the workdir as an artifact, for example `artifacts = ["/app"]`.
4. Grade `file_unchanged`, `git_clean`, and the side-effect contract against the
   baseline manifest in the verifier. Use git when `.git` survives.
5. Run remote-effect checks (`pushed`, `pr_created`) host-side after the job,
   from the pre/post `GitStateSnapshot`. Do not put a GitHub token in the
   verifier.
6. Reject `environment_mode = "shared"`. It weakens isolation and Harbor lint
   flags it (`shared-verifier`).

A `live-safe` or `live-external` case needs an allowlist and a short-lived,
least-privilege token in the agent sandbox. This is a deliberate exception. The
default is `no-network`.

The verifier must never run agent code on the host. `tests_pass` keeps its
current log-marker semantics in M2. A new `command_passes` assertion in M3 may
run tests inside the verifier.

### Q4. Deterministic grading in the verifier, LLM judge on the host

The verifier grades workspace state. The importer grades output-derived state.

| Check | Where | Input |
| --- | --- | --- |
| `file_exists`, `file_not_exists` | Verifier | workspace artifact |
| `file_contains` with `path` | Verifier | workspace artifact |
| `file_unchanged` | Verifier | artifact + baseline manifest |
| `git_clean` | Verifier | artifact + git, or manifest |
| `json_path` with `path` | Verifier | workspace artifact |
| side-effect contract | Verifier | artifact + baseline manifest |
| `command_ran` | Importer | ATIF tool calls |
| `file_contains` without `path` | Importer | agent transcript from ATIF |
| `json_path` without `path` | Importer | agent transcript from ATIF |
| `tests_pass` | Importer | raw agent log markers |
| trigger state | Importer | ATIF |
| remote effects | Importer | host `gh`/`glab` + snapshots |
| LLM rubric | Importer | workspace artifact + our `LLMGrader` |
| cost and time | Importer | `result.json` |

Reasons for the host-side judge:

- no network or API key in the verifier,
- changing the judge model does not rerun agents,
- the judge model is a normal suite setting,
- our report already owns scoring.

Consequence: Harbor's `reward.json` holds deterministic results only. The
`agent-skill-eval` report is the product. The importer writes `evals_meta.json`,
`outputs/output.txt`, and the artifact workspace path, so `grade` can change the
judge without rerunning agents. It re-runs deterministic assertions as well;
use the saved `grading.json` when only the judge changed.

The judge model is a per-suite field. Warn when the judge and the agent share a
provider or family. The current default judge and the proof-run agent are both
DeepSeek. That is a biased judge.

### Q5. Version pinning and CI

- Add an optional extra: `agent-skill-eval[harbor]` with `harbor==0.26.*`.
- Lazy import Harbor. Fail fast when the installed minor is not 0.26.
- Do not import Harbor internals. Use `result.json`, `config.json`,
  `lock.json`, `artifacts/`, and `agent/trajectory.json`.
- Record `harbor.version` from `lock.json` in `summary.json`.
- CI levels:
  1. Unit tests against recorded job directories. No Docker.
  2. `harbor lint` on exported tasks. No Docker.
  3. `harbor run --dry-run` on one exported task. No Docker.
  4. Weekly or manual: `harbor run` with the `oracle` agent in Docker. No LLM
     cost.
- `uv.lock` is gitignored. Track it or pin the extra exactly.

## M0 spike

Command shape:

```bash
harbor run -p <task> -a opencode-v2 \
  -m openrouter/deepseek/deepseek-v4.1-flash \
  --skill <skill> --job-name <arm> -o <jobs> -n 1
```

The task asked for `/app/hello.txt` with `Hello, world!`. The verifier used a
separate environment with `no-network`. The agent phase used `no-network` with
Harbor's automatic inference-host allowlist.

| Arm | Trials | Reward | Input tokens | Output tokens | Cost (USD) | Wall time |
| --- | --- | --- | --- | --- | --- | --- |
| with skill | 1 | 1.0 | 18,331 | 172 | 0.001386 | 5m 28s |
| without skill | 1 | 1.0 | 17,903 | 120 | 0.000872 | 4m 21s |

Both trials had `reward_valid=true` and no exceptions. The with-skill arm loaded
the skill. ATIF recorded a `skill` tool call and a `<skill_content
name="spike-hello">` observation.

Limitations of the spike:

- One trivial case cannot measure skill effect. Both arms passed.
- The skill directory name (`skill`) differed from the frontmatter name
  (`spike-hello`). Trigger detection must use the observation name.
- The task used a local Dockerfile and a separate verifier that reuses the agent
  environment. It did not use a dedicated verifier image.

## Limits

- ATIF is written into the agent's log directory. A hostile agent can write
  there. Trigger state and out-of-workspace access are signals, not scores.
  The importer checks the trajectory's agent name, but it cannot prove the file
  is untampered. Do not use these numbers as a pass/fail gate.
- Remote side-effect checks (`pushed`, `pr_created`, and the remote rules of a
  side-effect contract) run on the host, not in the verifier. With
  `--source-repo`, the host snapshots remote branch heads (`git ls-remote`) and
  PRs (`gh pr list`) before and after the jobs. A trial gets a new branch only
  when its name appears in the trial's commands or output; a PR counts when its
  head branch is one of those. Concurrent trials on one repo are therefore
  separated by branch name. Without `--source-repo`, these checks fail closed.
  The verifier holds no GitHub token.
- A dedicated verifier image must contain the grader package. The exporter
  writes `tests/Dockerfile` with the package copied into `/tests`.

## Findings for upstream

1. OpenCode V2 ATIF loses the `skill` tool input. `arguments` is `{"id":
   "skill"}`; the skill name is only in the observation. A reader that matches
   on arguments cannot detect the skill.
2. `tasks/separate-verifier.md` suggests `artifacts = ["/logs/agent/trajectory.json"]`
   for trajectory grading. `src/harbor/lint/task.py:_artifact_overlaps_logs`
   flags any artifact under `/logs/agent`, and `artifact-log-overlap` is a
   default lint rule.
3. Harbor's error policy marks `NonZeroAgentExitCodeError` and
   `AgentTimeoutError` as `has_valid_reward=true`. That is the false-zero case
   from the proof run. We override these to `has_valid_reward=false`.
4. `--diff` only reuses or regrades trials when the task has a semantic
   `version`. Without it, every trial reruns.
5. The default agent setup timeout is 360 seconds. Installing an agent CLI
   (Node via nvm plus npm) exceeded it in one M1 run. Set
   `agent_setup_timeout_multiplier` in the job config.
6. Harbor 0.26 requires Python 3.12 or newer. The `harbor` extra uses a
   `python_version >= '3.12'` marker so the native runner keeps Python 3.11
   support.
7. Git refuses a workdir owned by another uid (`dubious ownership`). The
   verifier sets `safe.directory` for the restored workdir. The verifier
   container is isolated, so this is safe.

## Implementation plan

1. M0: this ADR.
2. M1: `export-harbor` with golden tests and `harbor lint`.
3. M2: `run --backend harbor` and import from recorded job directories.
4. M3: tags, pass^k with Wilson intervals, trigger split, cost/time percentiles,
   baseline and compare, out-of-workspace metric, run sanity.
5. M4: README and ROADMAP.

One PR per milestone.
