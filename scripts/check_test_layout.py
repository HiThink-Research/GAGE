from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path


ALLOWED_TOP_LEVEL = {
    "__init__.py",
    "__pycache__",
    "_support",
    "conftest.py",
    "data",
    "e2e",
    "fixtures",
    "integration",
    "unit",
}


def _check_top_level_layout(tests_dir: Path) -> list[str]:
    errors: list[str] = []
    for entry in sorted(tests_dir.iterdir(), key=lambda item: item.name):
        if entry.name not in ALLOWED_TOP_LEVEL:
            errors.append(f"unexpected top-level tests entry: {entry.relative_to(tests_dir.parent)}")
    return errors


def _check_duplicate_basenames(tests_dir: Path) -> list[str]:
    basenames: dict[str, list[Path]] = defaultdict(list)
    for path in tests_dir.rglob("test_*.py"):
        if "__pycache__" in path.parts:
            continue
        basenames[path.name].append(path)

    errors: list[str] = []
    for name, paths in sorted(basenames.items()):
        if len(paths) < 2:
            continue
        rel_paths = ", ".join(sorted(str(path.relative_to(tests_dir.parent)) for path in paths))
        errors.append(f"duplicate test basename {name}: {rel_paths}")
    return errors


def _check_support_tree(tests_dir: Path) -> list[str]:
    errors: list[str] = []
    support_dir = tests_dir / "_support"
    if not support_dir.exists():
        return errors

    for path in support_dir.rglob("test_*.py"):
        if "__pycache__" in path.parts:
            continue
        errors.append(f"support module must not use test_ prefix: {path.relative_to(tests_dir.parent)}")
    return errors


def main() -> int:
    """Validate the repository test layout invariants."""

    repo_root = Path(__file__).resolve().parents[1]
    tests_dir = repo_root / "tests"
    errors: list[str] = []

    # STEP 1: Validate top-level directory ownership
    errors.extend(_check_top_level_layout(tests_dir))

    # STEP 2: Validate shared support placement
    errors.extend(_check_support_tree(tests_dir))

    # STEP 3: Detect duplicated pytest module names
    errors.extend(_check_duplicate_basenames(tests_dir))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("test layout check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
