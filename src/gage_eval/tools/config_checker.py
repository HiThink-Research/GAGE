"""Utility CLI that validates PipelineConfig YAML files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import yaml

from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.evaluation.task_plan import build_task_plan_specs


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config '{path}' must be a mapping at the top level")
    return data


def validate_config(path: Path, *, materialize_runtime: bool = False) -> None:
    payload = _load_yaml(path)
    config = PipelineConfig.from_dict(payload)
    build_task_plan_specs(config)
    if materialize_runtime:
        from gage_eval.config import build_default_registry
        from gage_eval.evaluation.runtime_builder import build_runtime
        from gage_eval.observability.trace import ObservabilityTrace
        from gage_eval.role.resource_profile import NodeResource, ResourceProfile

        registry = build_default_registry()
        profile = ResourceProfile(nodes=[NodeResource(node_id="local", gpus=1, cpus=8)])
        trace = ObservabilityTrace()
        build_runtime(config, registry, profile, trace=trace)
    print(f"[gage-eval] ✓ {path}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate gage-eval pipeline configs.")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Path to the YAML config file (can be supplied multiple times).",
    )
    parser.add_argument(
        "--materialize-runtime",
        action="store_true",
        help="Attempt to build runtime objects to catch registry/materialization issues.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        for item in args.config:
            validate_config(Path(item).resolve(), materialize_runtime=args.materialize_runtime)
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"[gage-eval] config validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
