from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agent_skill_eval.harbor import SUPPORTED_HARBOR_MINOR, harbor_version, require_harbor
from agent_skill_eval.harbor.export import export_suite

FIXTURES = Path(__file__).parent / "fixtures"
EVALS = FIXTURES / "evals" / "format-json.json"
SKILL = FIXTURES / "skills" / "format-json"
GOLDEN = FIXTURES / "harbor" / "golden"
REPO = Path(__file__).parents[1]
EXAMPLE_EVALS = REPO / "examples" / "fix-failing-tests" / "evals" / "evals.json"
EXAMPLE_SKILL = REPO / "skills" / "fix-failing-tests"


def _export(tmp_path: Path):
    return export_suite(evals_path=EVALS, skill_path=SKILL, out_dir=tmp_path / "out")


def test_harbor_version_guard():
    installed = harbor_version()
    if installed is None:
        with pytest.raises(RuntimeError, match="not installed"):
            require_harbor()
        return
    assert require_harbor() == installed
    assert installed.startswith(SUPPORTED_HARBOR_MINOR + ".")


def test_export_writes_one_task_per_arm(tmp_path: Path):
    manifest = _export(tmp_path)

    assert len(manifest.cases) == 3
    assert len(manifest.tasks) == 4
    assert manifest.tasks["case-explicit-invoke__with"] == "explicit-invoke"
    assert manifest.tasks["case-explicit-invoke__without"] == "explicit-invoke"
    assert manifest.task_arms["case-explicit-invoke__with"] == "with"
    assert manifest.task_arms["case-explicit-invoke__without"] == "without"
    assert manifest.task_arms["case-negative-control"] == "both"


def test_forced_instruction_only_changes_the_with_arm(tmp_path: Path):
    _export(tmp_path)
    tasks = tmp_path / "out" / "tasks"
    with_text = (tasks / "case-explicit-invoke__with" / "instruction.md").read_text()
    without_text = (tasks / "case-explicit-invoke__without" / "instruction.md").read_text()

    assert "Use the $format-json skill." in with_text
    assert "Use the $format-json skill." not in without_text
    assert "Format" in without_text


def test_graders_and_tests_never_enter_the_agent_environment(tmp_path: Path):
    _export(tmp_path)
    for task in (tmp_path / "out" / "tasks").iterdir():
        environment = task / "environment"
        assert not (environment / "tests").exists()
        assert not list(environment.rglob("test.sh"))
        assert not list(environment.rglob("case.json"))
        assert (task / "tests" / "case.json").is_file()
        assert (task / "tests" / "agent_skill_eval" / "graders" / "__init__.py").is_file()


def test_case_json_marks_output_assertions_deferred(tmp_path: Path):
    _export(tmp_path)
    tasks = tmp_path / "out" / "tasks"
    content = json.loads((tasks / "case-content-contains" / "tests" / "case.json").read_text())
    negative = json.loads((tasks / "case-negative-control" / "tests" / "case.json").read_text())

    assert content["deferred"] == [0]
    assert negative["deferred"] == []
    assert negative["should_trigger"] is False


def test_generated_files_match_golden(tmp_path: Path):
    _export(tmp_path)
    tasks = tmp_path / "out" / "tasks"
    pairs = [
        ("case-content-contains/task.toml", "content-contains/task.toml"),
        ("case-content-contains/instruction.md", "content-contains/instruction.md"),
        ("case-content-contains/environment/Dockerfile", "content-contains/environment/Dockerfile"),
        ("case-content-contains/tests/Dockerfile", "content-contains/tests/Dockerfile"),
        ("case-content-contains/tests/test.sh", "content-contains/tests/test.sh"),
        ("case-explicit-invoke__with/instruction.md", "explicit-with/instruction.md"),
    ]
    for produced, golden in pairs:
        assert (tasks / produced).read_text() == (GOLDEN / golden).read_text(), produced


def test_colliding_case_slugs_keep_every_case(tmp_path: Path):
    evals = tmp_path / "evals.json"
    evals.write_text(
        json.dumps(
            {
                "skill_name": "demo",
                "evals": [
                    {"id": "a b", "prompt": "one", "expected_output": "one"},
                    {"id": "a-b", "prompt": "two", "expected_output": "two"},
                ],
            }
        )
    )

    manifest = export_suite(evals_path=evals, skill_path=SKILL, out_dir=tmp_path / "out")

    assert len(manifest.tasks) == 2
    assert set(manifest.tasks.values()) == {"a b", "a-b"}
    assert len({path.name for path in (tmp_path / "out" / "tasks").iterdir()}) == 2


def test_forced_prefix_uses_the_frontmatter_name(tmp_path: Path):
    skill = tmp_path / "dir-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: front-skill\ndescription: Demo.\n---\n\nUse the front name.\n")
    evals = tmp_path / "evals.json"
    evals.write_text(
        json.dumps(
            {
                "skill_name": "dir-skill",
                "evals": [
                    {
                        "id": "forced",
                        "prompt": "Do the thing.",
                        "expected_output": "Done.",
                        "force_skill_invocation": True,
                    }
                ],
            }
        )
    )

    manifest = export_suite(evals_path=evals, skill_path=skill, out_dir=tmp_path / "out")

    assert manifest.skill_name == "dir-skill"
    assert manifest.skill_frontmatter_name == "front-skill"
    instruction = (tmp_path / "out" / "tasks" / "case-forced__with" / "instruction.md").read_text()
    assert "Use the $front-skill skill." in instruction
    assert "$dir-skill" not in instruction


def test_rejects_an_unsafe_workdir(tmp_path: Path):
    with pytest.raises(ValueError, match="workdir"):
        export_suite(
            evals_path=EVALS,
            skill_path=SKILL,
            out_dir=tmp_path / "out",
            workdir="/app; rm -rf /",
        )


def test_example_suite_exports_with_fixtures(tmp_path: Path):
    manifest = export_suite(
        evals_path=EXAMPLE_EVALS,
        skill_path=EXAMPLE_SKILL,
        out_dir=tmp_path / "example",
    )

    assert len(manifest.cases) == 4
    assert len(manifest.tasks) == 5
    task = tmp_path / "example" / "tasks" / "case-implicit-invoke"
    assert (task / "environment" / "fixtures" / "calculator.py").is_file()
    assert "COPY fixtures/ /app/" in (task / "environment" / "Dockerfile").read_text()

    pairs = [
        ("task.toml", "fix-failing-tests/task.toml"),
        ("instruction.md", "fix-failing-tests/instruction.md"),
        ("environment/Dockerfile", "fix-failing-tests/environment/Dockerfile"),
        ("tests/baseline.json", "fix-failing-tests/tests/baseline.json"),
    ]
    for produced, golden in pairs:
        assert (task / produced).read_text() == (GOLDEN / golden).read_text(), produced


@pytest.mark.skipif(shutil.which("harbor") is None, reason="harbor is not installed")
def test_exported_tasks_pass_harbor_lint(tmp_path: Path):
    _export(tmp_path)
    for task in sorted((tmp_path / "out" / "tasks").iterdir()):
        result = subprocess.run(
            ["harbor", "lint", str(task)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
