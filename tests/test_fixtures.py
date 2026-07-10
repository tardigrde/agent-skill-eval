from __future__ import annotations

from pathlib import Path

from agent_skill_eval.fixtures import build_fixture_file_map, fixture_target_collisions, workspace_fixture_path


class TestWorkspaceFixturePath:
    def test_direct_files_fixture_stages_at_workspace_root(self):
        assert workspace_fixture_path("files/config.json").as_posix() == "config.json"

    def test_single_case_fixture_folder_becomes_workspace_root(self):
        mapping = build_fixture_file_map(
            evals_dir=Path("/evals"),
            file_paths=[
                "files/prod-only/bootstrap/.smnow/change_template.json",
                "files/prod-only/security/env/prod/terraform.tfvars",
            ],
        )
        assert sorted(mapping) == [
            "bootstrap/.smnow/change_template.json",
            "security/env/prod/terraform.tfvars",
        ]

    def test_multiple_case_prefixes_are_preserved_to_avoid_collisions(self):
        mapping = build_fixture_file_map(
            evals_dir=Path("/evals"),
            file_paths=["files/dev/config.json", "files/prod/config.json"],
        )
        assert sorted(mapping) == ["dev/config.json", "prod/config.json"]

    def test_detects_workspace_target_collision(self):
        collisions = fixture_target_collisions(["files/prod/config.json", "files/config.json"])
        assert collisions == {"config.json": ["files/prod/config.json", "files/config.json"]}
