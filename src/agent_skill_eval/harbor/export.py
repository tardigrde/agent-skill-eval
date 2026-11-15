"""Export an eval suite as a directory of Harbor tasks."""

from __future__ import annotations

import shutil
from pathlib import Path

from agent_skill_eval.fixtures import build_fixture_file_map
from agent_skill_eval.graders import classify_assertion
from agent_skill_eval.harbor import harbor_version
from agent_skill_eval.harbor.baseline import build_baseline
from agent_skill_eval.harbor.models import HarborCase, HarborExportManifest
from agent_skill_eval.harbor.taskgen import (
    render_baseline_json,
    render_case_json,
    render_environment_dockerfile,
    render_harbor_case,
    render_instruction,
    render_manifest_json,
    render_task_toml,
    render_test_sh,
    render_tests_dockerfile,
    safe_task_name,
)
from agent_skill_eval.models import AssertionSpec, EvalCase, StructuredAssertion
from agent_skill_eval.skills import SkillInstaller
from agent_skill_eval.validation import load_and_validate_suite

DEFAULT_BASE_IMAGE = "python:3.12-slim"
DEFAULT_WORKDIR = "/app"
DEFAULT_TASK_VERSION = "1.0"
_DEFERRED_STRUCTURED = {"command_ran", "command_not_ran", "tests_pass"}
_DEFERRED_PROSE = {"pushed", "pr-created", "command-ran", "content-contains", "valid-json", "llm"}


def _structured(assertion: AssertionSpec) -> StructuredAssertion | None:
    return assertion if isinstance(assertion, StructuredAssertion) else None


def assertion_is_deferred(assertion: AssertionSpec) -> bool:
    """True when the verifier lacks the input this assertion needs.

    Output-derived checks run in the importer, where the agent transcript is
    available. Workspace checks run in the verifier.
    """
    structured = _structured(assertion)
    if structured is not None:
        if structured.type in _DEFERRED_STRUCTURED:
            return True
        if structured.type in {"file_contains", "file_not_contains", "json_path", "json_path_not_exists"}:
            return not structured.path
        return False
    return classify_assertion(assertion) in _DEFERRED_PROSE


def deferred_assertions(assertions: list[AssertionSpec]) -> list[int]:
    return [index for index, assertion in enumerate(assertions) if assertion_is_deferred(assertion)]


def _write(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _copy_fixtures(task_dir: Path, fixture_map: dict[str, Path]) -> bool:
    if not fixture_map:
        return False
    root = task_dir / "environment" / "fixtures"
    for target, source in sorted(fixture_map.items()):
        destination = root / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return True


def _copy_grader(task_tests_dir: Path) -> None:
    package_root = Path(__file__).resolve().parents[1]
    files = [
        "__init__.py",
        "models.py",
        "git_state.py",
        "graders/__init__.py",
        "harbor/__init__.py",
        "harbor/baseline.py",
        "harbor/models.py",
        "harbor/verifier.py",
    ]
    for relative in files:
        source = package_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Verifier dependency is missing: {source}")
        destination = task_tests_dir / "agent_skill_eval" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _task_variants(case: EvalCase, task_base: str) -> list[tuple[str, str, bool]]:
    if case.force_skill_invocation:
        return [
            (f"{task_base}__with", "with", True),
            (f"{task_base}__without", "without", False),
        ]
    return [(task_base, "both", False)]


def export_suite(
    *,
    evals_path: Path,
    skill_path: Path,
    out_dir: Path,
    base_image: str = DEFAULT_BASE_IMAGE,
    workdir: str = DEFAULT_WORKDIR,
    task_version: str = DEFAULT_TASK_VERSION,
    eval_ids: list[str] | None = None,
    side_effect_levels: list[str] | None = None,
    force: bool = False,
) -> HarborExportManifest:
    """Write one Harbor task per eval case and return the export manifest."""
    evals_path = evals_path.resolve()
    skill_path = skill_path.resolve()
    out_dir = out_dir.resolve()

    installer = SkillInstaller(skill_path)
    problems = installer.frontmatter_problems()
    if problems:
        raise ValueError("Skill is not usable: " + "; ".join(problems))
    skill_name = installer.skill_name
    frontmatter_name = installer.frontmatter_name()

    suite, _report = load_and_validate_suite(evals_path, eval_ids, side_effect_levels)

    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise ValueError(f"Output directory is not empty: {out_dir}. Use --force to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_root = out_dir / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)

    manifest = HarborExportManifest(
        suite=evals_path.stem,
        skill_name=skill_name,
        skill_frontmatter_name=frontmatter_name,
        skill_hash=installer.content_hash(),
        evals_path=str(evals_path),
        out_dir=str(out_dir),
        workdir=workdir,
        base_image=base_image,
        task_version=task_version,
        harbor_version=harbor_version(),
    )

    prefix_text = f"Use the ${skill_name} skill."

    for case in suite.evals:
        fixture_map = build_fixture_file_map(evals_path.parent, case.files)
        task_base = safe_task_name(str(case.id))
        task_names: list[str] = []
        tags = ["correctness"]

        for task_name, arm, needs_prefix in _task_variants(case, task_base):
            prefix = prefix_text if needs_prefix else None
            task_dir = tasks_root / task_name
            has_fixtures = _copy_fixtures(task_dir, fixture_map)
            baseline = build_baseline(fixture_map, workdir)
            live = case.side_effect_level in {"live-safe", "live-external"}
            deferred = deferred_assertions(case.assertions)

            _write(
                task_dir / "task.toml",
                render_task_toml(
                    case=case,
                    task_name=task_name,
                    workdir=workdir,
                    task_version=task_version,
                    live=live,
                ),
            )
            _write(task_dir / "instruction.md", render_instruction(case.prompt, prefix))
            _write(
                task_dir / "environment" / "Dockerfile",
                render_environment_dockerfile(
                    base_image=base_image,
                    workdir=workdir,
                    has_fixtures=has_fixtures,
                ),
            )
            tests_dir = task_dir / "tests"
            _write(tests_dir / "Dockerfile", render_tests_dockerfile(workdir=workdir))
            _write(tests_dir / "test.sh", render_test_sh(), executable=True)
            _write(
                tests_dir / "verifier.py",
                "from agent_skill_eval.harbor.verifier import main\n\n"
                'if __name__ == "__main__":\n'
                "    raise SystemExit(main())\n",
            )
            _write(
                tests_dir / "case.json",
                render_case_json(
                    case=case,
                    task_name=task_name,
                    workdir=workdir,
                    deferred=deferred,
                    tags=tags,
                    skill_name=skill_name,
                    skill_frontmatter_name=frontmatter_name,
                ),
            )
            _write(tests_dir / "baseline.json", render_baseline_json(baseline))
            _copy_grader(tests_dir)

            task_names.append(task_name)
            manifest.tasks[task_name] = str(case.id)
            manifest.task_arms[task_name] = arm

        manifest.cases.append(render_harbor_case(case, task_names, "both", tags))

    _write(out_dir / "manifest.json", render_manifest_json(manifest))
    return manifest


def load_manifest(out_dir: Path) -> HarborExportManifest:
    return HarborExportManifest.model_validate_json((out_dir / "manifest.json").read_text(encoding="utf-8"))


def case_by_id(manifest: HarborExportManifest, case_id: str) -> HarborCase | None:
    return next((case for case in manifest.cases if case.case_id == case_id), None)
