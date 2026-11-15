"""Harbor backend support.

This package is import-safe without Harbor installed. Harbor is an optional
extra; the native runner must keep working without Docker or Harbor.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

SUPPORTED_HARBOR_MINOR = "0.26"


def harbor_version() -> str | None:
    """Return the installed Harbor version, or None when it is not installed."""
    try:
        return version("harbor")
    except PackageNotFoundError:
        return None


def require_harbor() -> str:
    """Return the installed Harbor version, or raise a clear error.

    Harbor is a 0.x package with breaking changes in minor releases, so the
    guard checks the minor version and not only presence.
    """
    found = harbor_version()
    if found is None:
        raise RuntimeError(
            "Harbor is not installed. Install the optional extra: pip install 'agent-skill-eval[harbor]'."
        )
    if not found.startswith(SUPPORTED_HARBOR_MINOR + "."):
        raise RuntimeError(
            f"agent-skill-eval supports Harbor {SUPPORTED_HARBOR_MINOR}.x, "
            f"but harbor {found} is installed. Install the pinned minor: "
            f"pip install 'harbor=={SUPPORTED_HARBOR_MINOR}.*'."
        )
    return found
