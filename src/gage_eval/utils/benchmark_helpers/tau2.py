"""Shared helpers for Tau2 integration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


_TAU2_IMPORT_HINT = (
    "tau2 is not installed. Install the tau2-bench package (editable recommended) "
    "or set TAU2_DATA_DIR to point at the tau2-bench data directory."
)


def ensure_tau2_importable() -> None:
    """Ensure the tau2 package can be imported.

    Raises:
        RuntimeError: If tau2 cannot be imported.
    """

    try:
        import tau2  # noqa: F401
    except Exception as exc:
        raise RuntimeError(_TAU2_IMPORT_HINT) from exc


def resolve_tau2_data_dir(explicit: Optional[str] = None) -> Path:
    """Resolve the tau2 data directory.

    Priority order:
    1) explicit argument
    2) TAU2_DATA_DIR env var
    3) tau2.utils.utils.DATA_DIR (if tau2 is installed)

    Args:
        explicit: Explicit data directory path, if provided.

    Returns:
        Resolved Path to the tau2 data directory.

    Raises:
        FileNotFoundError: If no valid data directory can be resolved.
    """

    if explicit:
        return Path(explicit).expanduser().resolve()
    env_dir = os.environ.get("TAU2_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    try:
        from tau2.utils.utils import DATA_DIR  # type: ignore

        return Path(DATA_DIR).expanduser().resolve()
    except Exception as exc:
        raise FileNotFoundError(
            "TAU2_DATA_DIR is not configured and tau2 DATA_DIR fallback is unavailable. "
            "Set TAU2_DATA_DIR or install tau2-bench (editable) to use its bundled data."
        ) from exc


def load_tau2_tasks(path: Path) -> list[dict[str, Any]]:
    """Load tau2 tasks from a JSON file.

    Args:
        path: Path to tasks.json (or compatible JSON file).

    Returns:
        List of task dictionaries.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file content is invalid or not JSON.
    """

    if not path.exists():
        raise FileNotFoundError(f"Tau2 tasks file not found: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported tau2 tasks file type: {path.suffix}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        tasks = payload.get("tasks")
    else:
        tasks = payload
    if not isinstance(tasks, list):
        raise ValueError(f"Tau2 tasks file is malformed: {path}")
    return tasks


def load_tau2_split(path: Path) -> Optional[dict[str, list[str]]]:
    """Load tau2 task split mapping if available."""

    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Tau2 split file is malformed: {path}")
    return {str(k): list(v) for k, v in payload.items() if isinstance(v, list)}

