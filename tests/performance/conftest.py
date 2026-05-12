from __future__ import annotations

from pathlib import Path
from typing import Any


def pytest_ignore_collect(collection_path: Path, config: Any) -> bool:
    """Exclude performance baselines unless the performance tree is requested explicitly."""

    performance_root = Path(__file__).resolve().parent
    requested_paths = [
        Path(str(arg)).resolve()
        for arg in config.args
        if arg and not str(arg).startswith("-")
    ]
    if any(path == performance_root or performance_root in path.parents for path in requested_paths):
        return False
    return performance_root in Path(collection_path).resolve().parents
