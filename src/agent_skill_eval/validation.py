from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent_skill_eval.fixtures import fixture_target_collisions, unsafe_fixture_path_reason
from agent_skill_eval.graders import assertion_text, classify_assertion
from agent_skill_eval.models import EvalSuite, StructuredAssertion


class EvalSuiteValidationError(ValueError):
    """Raised when an eval suite is structurally invalid."""


@dataclass(frozen=True)
class AssertionClassification:
    eval_id: str
    assertion: str
    method: str


@dataclass(frozen=True)
class EvalSuiteValidationReport:
    evals_path: Path
    skill_name: str
    selected_eval_count: int
    selected_by_eval_id: bool
    total_assertions: int
    side_effect_counts: dict[str, int]
    selected_side_effect_levels: tuple[str, ...]
    assertion_classifications: tuple[AssertionClassification, ...]
    strict_warnings: tuple[str, ...] = ()

    @property
    def llm_assertions(self) -> tuple[AssertionClassification, ...]:
        return tuple(a for a in self.assertion_classifications if a.method == "llm")

    @property
    def deterministic_assertions(self) -> tuple[AssertionClassification, ...]:
        return tuple(a for a in self.assertion_classifications if a.method != "llm")


def load_and_validate_suite(
    evals_path: Path,
    eval_ids: list[str] | None = None,
    side_effect_levels: list[str] | None = None,
) -> tuple[EvalSuite, EvalSuiteValidationReport]:
    """Load an eval suite and apply all pre-run validation checks.

    This is intentionally shared by ``validate`` and ``run`` so a suite that
    passes the cheap validation command cannot later fail differently, or
    worse, silently skip missing fixtures during a live agent run.
    """

    if not evals_path.exists():
        raise EvalSuiteValidationError(f"File not found: {evals_path}")

    try:
        data = json.loads(evals_path.read_text())
    except json.JSONDecodeError as e:
        raise EvalSuiteValidationError(f"Invalid JSON: {e}") from e

    try:
        full_suite = EvalSuite(**data)
    except Exception as e:
        raise EvalSuiteValidationError(f"Schema validation failed:\n{e}") from e

    ids = [str(e.id) for e in full_suite.evals]
    duplicates = sorted({eval_id for eval_id in ids if ids.count(eval_id) > 1})
    if duplicates:
        raise EvalSuiteValidationError(f"Duplicate eval ids: {duplicates}")

    try:
        suite = full_suite.filtered_by_ids(eval_ids)
    except ValueError as e:
        raise EvalSuiteValidationError(str(e)) from e

    try:
        suite = suite.filtered_by_side_effect_levels(side_effect_levels)
    except ValueError as e:
        raise EvalSuiteValidationError(str(e)) from e
    if side_effect_levels and not suite.evals:
        raise EvalSuiteValidationError(
            "No evals matched side-effect level filter. Requested: " + ", ".join(side_effect_levels)
        )

    fixture_errors = []
    missing_files = []
    for eval_case in suite.evals:
        for file_path in eval_case.files:
            reason = unsafe_fixture_path_reason(file_path)
            if reason:
                fixture_errors.append(f"eval '{eval_case.id}': {file_path}: {reason}")
        collisions = fixture_target_collisions(eval_case.files)
        for target, sources in collisions.items():
            fixture_errors.append(
                f"eval '{eval_case.id}': workspace fixture target {target!r} is used by: " + ", ".join(sources)
            )
        for file_path in eval_case.files:
            if not (evals_path.parent / file_path).exists():
                missing_files.append(f"eval '{eval_case.id}': {file_path}")
    if fixture_errors:
        raise EvalSuiteValidationError("Fixture path problems:\n" + "\n".join(f"  - {m}" for m in fixture_errors))
    if missing_files:
        raise EvalSuiteValidationError(
            "Referenced fixture files not found:\n" + "\n".join(f"  - {m}" for m in missing_files)
        )

    structured_errors = _structured_assertion_errors(suite)
    if structured_errors:
        raise EvalSuiteValidationError(
            "Structured assertion problems:\n" + "\n".join(f"  - {m}" for m in structured_errors)
        )

    classifications: list[AssertionClassification] = []
    total_assertions = 0
    side_effect_counts: dict[str, int] = {}
    for eval_case in suite.evals:
        side_effect_counts[eval_case.side_effect_level] = side_effect_counts.get(eval_case.side_effect_level, 0) + 1
        for assertion in eval_case.assertions:
            total_assertions += 1
            classifications.append(
                AssertionClassification(
                    eval_id=str(eval_case.id),
                    assertion=assertion_text(assertion),
                    method=classify_assertion(assertion),
                )
            )

    report = EvalSuiteValidationReport(
        evals_path=evals_path,
        skill_name=suite.skill_name,
        selected_eval_count=len(suite.evals),
        selected_by_eval_id=bool(eval_ids),
        total_assertions=total_assertions,
        side_effect_counts=side_effect_counts,
        selected_side_effect_levels=tuple(side_effect_levels or []),
        assertion_classifications=tuple(classifications),
        strict_warnings=tuple(_strict_warnings(suite, classifications)),
    )
    return suite, report


def _structured_assertion_errors(suite: EvalSuite) -> list[str]:
    errors: list[str] = []
    for eval_case in suite.evals:
        for index, assertion in enumerate(eval_case.assertions, start=1):
            if not isinstance(assertion, StructuredAssertion):
                continue
            prefix = f"eval '{eval_case.id}' assertion {index} ({assertion.type})"
            for field in _required_structured_fields(assertion):
                value = getattr(assertion, field)
                if value is None or value == "":
                    errors.append(f"{prefix}: missing required field '{field}'")
            if assertion.path:
                path = Path(assertion.path)
                if path.is_absolute() or ".." in path.parts:
                    errors.append(f"{prefix}: path must be relative and stay inside the workspace")
    return errors


def _required_structured_fields(assertion: StructuredAssertion) -> tuple[str, ...]:
    if assertion.type in {"file_exists", "file_not_exists", "file_unchanged"}:
        return ("path",)
    if assertion.type in {"file_contains", "file_not_contains"}:
        return ("contains",)
    if assertion.type in {"command_ran", "command_not_ran"}:
        return ("command",)
    if assertion.type in {"json_path", "json_path_not_exists"}:
        return ("json_path",)
    return ()


def _strict_warnings(suite: EvalSuite, classifications: list[AssertionClassification]) -> list[str]:
    warnings: list[str] = []
    if not suite.evals:
        warnings.append("suite has no selected eval cases")
        return warnings

    if not classifications:
        warnings.append("suite has no assertions")
    elif all(item.method == "llm" for item in classifications):
        warnings.append("all assertions require LLM grading; add deterministic assertions for CI-grade signal")

    if all(eval_case.force_skill_invocation for eval_case in suite.evals):
        warnings.append("all evals force skill invocation; add at least one natural trigger case")
    if not any(eval_case.should_trigger and not eval_case.force_skill_invocation for eval_case in suite.evals):
        warnings.append("suite has no natural trigger case")
    if not _has_negative_control(suite):
        warnings.append("suite has no negative or missing-input control case")

    levels = {eval_case.side_effect_level for eval_case in suite.evals}
    if not (levels & {"static", "local-only"}):
        warnings.append("suite has no static or local-only eval; add a cheap pre-flight case")
    return warnings


def _has_negative_control(suite: EvalSuite) -> bool:
    negative_types = {"file_not_exists", "file_not_contains", "command_not_ran", "json_path_not_exists"}
    negative_id_words = (
        "missing",
        "negative",
        "invalid",
        "without",
        "should-not",
        "does-not",
        "no-",
        "stop",
    )
    negative_text = ("does not", "do not", "must not", "should not", "not execute", "not write", "missing")
    for eval_case in suite.evals:
        eval_id = str(eval_case.id).lower()
        if not eval_case.should_trigger:
            return True
        if any(word in eval_id for word in negative_id_words):
            return True
        for assertion in eval_case.assertions:
            if isinstance(assertion, StructuredAssertion) and assertion.type in negative_types:
                return True
            text = assertion_text(assertion).lower()
            if any(phrase in text for phrase in negative_text):
                return True
    return False
