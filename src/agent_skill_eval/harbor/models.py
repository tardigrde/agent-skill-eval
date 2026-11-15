"""Models for the Harbor export manifest and baseline snapshot."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Arm = Literal["with", "without", "both"]


class HarborBaselineFile(BaseModel):
    path: str
    sha256: str
    mode: int
    size: int


class HarborBaseline(BaseModel):
    schema_version: int = 1
    workdir: str = "/app"
    tree_hash: Optional[str] = None
    branch: str = "main"
    files: list[HarborBaselineFile] = Field(default_factory=list)
    dirs: list[str] = Field(default_factory=list)


class HarborCase(BaseModel):
    case_id: str
    task_names: list[str] = Field(default_factory=list)
    arm: Arm = "both"
    tags: list[str] = Field(default_factory=list)
    should_trigger: bool = True
    force_skill_invocation: bool = False
    side_effect_level: str = "local-only"


class HarborExportManifest(BaseModel):
    schema_version: int = 1
    suite: str
    skill_name: str
    skill_frontmatter_name: str = ""
    skill_hash: str
    evals_path: str
    out_dir: str
    workdir: str
    base_image: str
    task_version: str
    harbor_version: Optional[str] = None
    cases: list[HarborCase] = Field(default_factory=list)
    tasks: dict[str, str] = Field(default_factory=dict)
    task_arms: dict[str, str] = Field(default_factory=dict)

    def case_for_task(self, task_name: str) -> HarborCase | None:
        case_id = self.tasks.get(task_name)
        if case_id is None:
            return None
        return next((case for case in self.cases if case.case_id == case_id), None)
