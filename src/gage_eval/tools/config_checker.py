"""Utility CLI that validates PipelineConfig YAML files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import yaml

from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.evaluation.task_plan import build_task_plan_specs
from gage_eval.tools.distill import calculate_definition_digest


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config '{path}' must be a mapping at the top level")
    return data


def _validate_builtin_template(payload: dict, *, materialize_runtime: bool = False) -> None:
    """Validate BuiltinTemplate by reusing PipelineConfig schema on its definition block."""

    meta = payload.get("metadata") or {}
    definition = payload.get("definition")
    if not isinstance(definition, dict):
        raise ValueError("BuiltinTemplate must contain a 'definition' mapping")
    if not meta.get("name"):
        raise ValueError("BuiltinTemplate.metadata.name is required")
    if not meta.get("version"):
        raise ValueError("BuiltinTemplate.metadata.version is required")

    # 校验 digest（如果存在）与 definition 一致性，帮助早期发现改动未更新的问题。
    digest = meta.get("digest")
    if isinstance(digest, str) and digest.startswith("sha256:"):
        expected = digest.split("sha256:", 1)[1]
        computed = calculate_definition_digest(definition)
        if computed != expected:
            raise ValueError(
                f"BuiltinTemplate digest mismatch: metadata {expected} vs computed {computed}. "
                "Please recompute digest after modifying definition."
            )

    pipeline_payload = {"api_version": "gage/v1alpha1", "kind": "PipelineConfig"}
    pipeline_payload.update(definition)
    config = PipelineConfig.from_dict(pipeline_payload)
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


def validate_config(path: Path, *, materialize_runtime: bool = False) -> None:
    payload = _load_yaml(path)
    kind = (payload.get("kind") or "").lower()
    if kind in {"builtintemplate", "builtin"}:
        _validate_builtin_template(payload, materialize_runtime=materialize_runtime)
        print(f"[gage-eval] ✓ (BuiltinTemplate) {path}")
        return

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
