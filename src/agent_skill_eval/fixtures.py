from __future__ import annotations

from pathlib import Path


def build_fixture_file_map(evals_dir: Path, file_paths: list[str]) -> dict[str, Path]:
    """Map eval fixture paths to workspace-relative target paths.

    Fixtures usually live under ``evals/files``. For case-scoped fixture
    folders such as ``files/sparse/README.md``, the case folder becomes the
    workspace root for that eval, preserving current README-oriented suites.
    If one eval references multiple case folders, the folder names are kept
    so same-named files cannot silently overwrite each other.
    """
    strip_case_prefix = _single_case_prefix(file_paths)
    fixture_files: dict[str, Path] = {}
    for file_path in file_paths:
        target = workspace_fixture_path(file_path, strip_case_prefix=strip_case_prefix)
        fixture_files[target.as_posix()] = evals_dir / file_path
    return fixture_files


def fixture_target_collisions(file_paths: list[str]) -> dict[str, list[str]]:
    strip_case_prefix = _single_case_prefix(file_paths)
    targets: dict[str, list[str]] = {}
    for file_path in file_paths:
        target = workspace_fixture_path(file_path, strip_case_prefix=strip_case_prefix).as_posix()
        targets.setdefault(target, []).append(file_path)
    return {target: sources for target, sources in targets.items() if len(sources) > 1}


def unsafe_fixture_path_reason(file_path: str) -> str | None:
    path = Path(file_path)
    if not file_path.strip():
        return "path is empty"
    if path.is_absolute():
        return "path must be relative"
    if ".." in path.parts:
        return "path must stay inside the evals directory"
    return None


def workspace_fixture_path(file_path: str, strip_case_prefix: str | None = None) -> Path:
    path = Path(file_path)
    parts = path.parts
    if len(parts) >= 2 and parts[0] == "files":
        if len(parts) == 2:
            return Path(parts[1])
        if strip_case_prefix and parts[1] == strip_case_prefix:
            return Path(*parts[2:])
        return Path(*parts[1:])
    return path


def _single_case_prefix(file_paths: list[str]) -> str | None:
    prefixes = {
        Path(file_path).parts[1]
        for file_path in file_paths
        if len(Path(file_path).parts) > 2 and Path(file_path).parts[0] == "files"
    }
    if len(prefixes) == 1:
        return next(iter(prefixes))
    return None
