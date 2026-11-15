"""Render Harbor task files.

The output is intentionally plain text. Golden tests compare the generated
files byte for byte, so avoid nondeterministic values (timestamps, absolute
paths) in the templates.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from agent_skill_eval.harbor.models import HarborBaseline, HarborCase
from agent_skill_eval.models import AssertionSpec, EvalCase

VERIFIER_PYDANTIC_VERSION = "2.13.4"
LIVE_HOSTS = ("github.com", "api.github.com")
_BASE_PACKAGES = "git ca-certificates curl bash"


def toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def toml_list(values: Iterable[str]) -> str:
    return "[" + ", ".join(toml_string(value) for value in values) + "]"


def safe_task_name(case_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(case_id)).strip("._-")
    if not slug:
        slug = "case"
    return f"case-{slug[:48]}"


def assertion_payload(assertions: list[AssertionSpec]) -> list[Any]:
    payload: list[Any] = []
    for assertion in assertions:
        if isinstance(assertion, str):
            payload.append(assertion)
        else:
            payload.append(assertion.model_dump(mode="json", exclude_none=True))
    return payload


def render_instruction(prompt: str, force_prefix: str | None = None) -> str:
    text = prompt
    if force_prefix:
        text = f"{force_prefix} {prompt}".strip()
    return text.rstrip() + "\n"


def render_task_toml(
    *,
    case: EvalCase,
    task_name: str,
    workdir: str,
    task_version: str,
    live: bool,
) -> str:
    lines = [
        'schema_version = "1.4"',
        f"version = {toml_string(task_version)}",
        f"artifacts = {toml_list([workdir])}",
        "",
        "[task]",
        f"name = {toml_string('skill-eval/' + task_name)}",
        f"description = {toml_string('skill-eval case ' + str(case.id))}",
        'authors = [{ name = "agent-skill-eval" }]',
        "",
        "[metadata.skill_eval]",
        f"case_id = {toml_string(str(case.id))}",
        f"tags = {toml_list(['correctness'])}",
        f"should_trigger = {str(case.should_trigger).lower()}",
        f"force_skill_invocation = {str(case.force_skill_invocation).lower()}",
        f"side_effect_level = {toml_string(case.side_effect_level)}",
        f"workdir = {toml_string(workdir)}",
        "",
        "[verifier]",
        'environment_mode = "separate"',
        "timeout_sec = 600.0",
        "",
        "[verifier.environment]",
        'network_mode = "no-network"',
        "",
        "[agent]",
        "timeout_sec = 600.0",
        'network_mode = "allowlist"' if live else 'network_mode = "no-network"',
    ]
    if live:
        lines.append(f"allowed_hosts = {toml_list(LIVE_HOSTS)}")
    lines.extend(
        [
            "",
            "[environment]",
            'network_mode = "public"',
            "cpus = 2",
            "memory_mb = 4096",
            f"workdir = {toml_string(workdir)}",
        ]
    )
    if live:
        lines.extend(
            [
                "",
                "[environment.env]",
                'GH_TOKEN = "${GH_TOKEN}"',
                'GITHUB_TOKEN = "${GH_TOKEN}"',
            ]
        )
    return "\n".join(lines) + "\n"


def render_environment_dockerfile(
    *,
    base_image: str,
    workdir: str,
    has_fixtures: bool,
) -> str:
    lines = [
        f"FROM {base_image}",
        "",
        "RUN if command -v apt-get >/dev/null 2>&1; then \\",
        f"      apt-get update && apt-get install -y --no-install-recommends {_BASE_PACKAGES} \\",
        "      && rm -rf /var/lib/apt/lists/*; \\",
        "    elif command -v apk >/dev/null 2>&1; then \\",
        f"      apk add --no-cache {_BASE_PACKAGES}; \\",
        "    fi",
        "",
        f"RUN mkdir -p {workdir}",
        f"WORKDIR {workdir}",
    ]
    if has_fixtures:
        lines.extend(["", f"COPY fixtures/ {workdir}/"])
    lines.extend(
        [
            "",
            f"RUN git init -q -b main {workdir} \\",
            f'    && git -C {workdir} config user.email "eval@agent-skill-eval.local" \\',
            f'    && git -C {workdir} config user.name "agent-skill-eval" \\',
            f"    && git -C {workdir} add -A \\",
            '    && GIT_AUTHOR_DATE="2000-01-01T00:00:00+0000" \\',
            '       GIT_COMMITTER_DATE="2000-01-01T00:00:00+0000" \\',
            f'       git -C {workdir} commit -q --allow-empty -m "baseline" \\',
            f"    && git -C {workdir} tag baseline \\",
            f"    && git -C {workdir} rev-parse HEAD > {workdir}/.git/eval-baseline",
        ]
    )
    return "\n".join(lines) + "\n"


def render_tests_dockerfile(*, workdir: str) -> str:
    return "\n".join(
        [
            "FROM python:3.12-slim",
            "",
            "RUN apt-get update \\",
            f"    && apt-get install -y --no-install-recommends {_BASE_PACKAGES} \\",
            "    && rm -rf /var/lib/apt/lists/*",
            "",
            f"RUN pip install --no-cache-dir pydantic=={VERIFIER_PYDANTIC_VERSION}",
            "",
            f"WORKDIR {workdir}",
            "COPY . /tests/",
            "",
        ]
    )


def render_test_sh() -> str:
    return "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "",
            "mkdir -p /logs/verifier",
            "export PYTHONPATH=/tests",
            "python /tests/verifier.py",
            "",
        ]
    )


def render_case_json(
    *,
    case: EvalCase,
    task_name: str,
    workdir: str,
    deferred: list[int],
    tags: list[str],
    skill_name: str,
    skill_frontmatter_name: str,
) -> str:
    payload = {
        "schema_version": 1,
        "case_id": str(case.id),
        "task_name": task_name,
        "workdir": workdir,
        "assertions": assertion_payload(case.assertions),
        "deferred": deferred,
        "expected_output": case.expected_output,
        "should_trigger": case.should_trigger,
        "force_skill_invocation": case.force_skill_invocation,
        "side_effect_contract": (
            case.side_effect_contract.model_dump(mode="json") if case.side_effect_contract else None
        ),
        "side_effect_level": case.side_effect_level,
        "tags": tags,
        "skill_name": skill_name,
        "skill_frontmatter_name": skill_frontmatter_name,
    }
    return json.dumps(payload, indent=2) + "\n"


def render_baseline_json(baseline: HarborBaseline) -> str:
    return baseline.model_dump_json(indent=2) + "\n"


def render_manifest_json(manifest: Any) -> str:
    return manifest.model_dump_json(indent=2) + "\n"


def render_harbor_case(case: EvalCase, task_names: list[str], arm: str, tags: list[str]) -> HarborCase:
    return HarborCase(
        case_id=str(case.id),
        task_names=task_names,
        arm=arm,  # type: ignore[arg-type]
        tags=tags,
        should_trigger=case.should_trigger,
        force_skill_invocation=case.force_skill_invocation,
        side_effect_level=case.side_effect_level,
    )
