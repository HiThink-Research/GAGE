"""Helpers for loading and materializing configuration payloads."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, TypeAlias

import yaml

from gage_eval.config.loader_cli import CLIIntent, apply_cli_final_overrides
from gage_eval.config.schema import normalize_pipeline_payload
from gage_eval.config.smart_defaults.registry import apply_smart_defaults
from gage_eval.config.smart_defaults.profiles import select_smart_defaults_profile
from gage_eval.config.smart_defaults.types import RuleContext

RunConfigCompiler: TypeAlias = Callable[[dict[str, Any]], tuple[dict[str, Any], Path | None]]

_ENV_PATTERN = re.compile(r"^\$\{([^}:]+)(?:(:-|:\?)(.*))?\}$")
_MAX_RUNCONFIG_DEPTH = 32


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at the top level."""

    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' not found")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config '{path}' must be a mapping at the top level")
    return data


def expand_env(value: Any) -> Any:
    """Recursively expand whole-string environment placeholders in a payload."""

    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, str):
        match = _ENV_PATTERN.match(value)
        if not match:
            return value
        var, operator, operand = match.group(1), match.group(2), match.group(3)
        if operator == ":?" and not os.getenv(var):
            raise ValueError(operand or f"environment variable {var} is required")
        resolved = os.getenv(var, operand if operator == ":-" and operand is not None else "")
        if isinstance(resolved, str) and resolved.strip():
            try:
                return int(resolved)
            except ValueError:
                try:
                    return float(resolved)
                except ValueError:
                    return resolved
        return resolved
    return value


def materialize_pipeline_config_payload(
    payload: dict[str, Any],
    source_path: Path | None,
    run_config_compiler: RunConfigCompiler | None = None,
    *,
    cli_intent: CLIIntent | None = None,
    smart_defaults: bool = True,
) -> dict[str, Any]:
    """Expand env vars, compile RunConfig payloads, and normalize PipelineConfig payloads."""

    materialized = _materialize_payload(payload, source_path, run_config_compiler)
    intent = cli_intent or CLIIntent()
    if smart_defaults:
        materialized = _apply_smart_defaults(materialized, source_path, intent)
    apply_cli_final_overrides(materialized, intent)
    materialized.pop("scene", None)
    return normalize_pipeline_payload(materialized)


def load_pipeline_config_payload(
    path: Path,
    run_config_compiler: RunConfigCompiler | None = None,
    *,
    cli_intent: CLIIntent | None = None,
    smart_defaults: bool = True,
) -> dict[str, Any]:
    """Load a PipelineConfig payload from YAML and fully materialize it."""

    return materialize_pipeline_config_payload(
        load_yaml_mapping(path),
        path,
        run_config_compiler,
        cli_intent=cli_intent,
        smart_defaults=smart_defaults,
    )


def load_pre_smart_defaults_payload(
    path: Path,
    run_config_compiler: RunConfigCompiler | None = None,
) -> dict[str, Any]:
    """Load a payload for pre-smart-defaults flows without schema normalization."""

    materialized = _materialize_payload(load_yaml_mapping(path), path, run_config_compiler)
    select_smart_defaults_profile(materialized, path)
    return materialized


def _materialize_payload(
    payload: dict[str, Any],
    source_path: Path | None,
    run_config_compiler: RunConfigCompiler | None,
    _depth: int = 0,
) -> dict[str, Any]:
    materialized = expand_env(deepcopy(payload))
    kind = str(materialized.get("kind") or "")
    if kind.lower() == "runconfig":
        if run_config_compiler is None:
            raise ValueError("RunConfig payload requires run_config_compiler")
        if _depth >= _MAX_RUNCONFIG_DEPTH:
            source_hint = f" for '{source_path}'" if source_path is not None else ""
            raise ValueError(
                f"RunConfig materialization exceeded {_MAX_RUNCONFIG_DEPTH} nested compilations{source_hint}"
            )
        compiled_payload, template_path = run_config_compiler(deepcopy(materialized))
        next_source_path = template_path or source_path
        return _materialize_payload(
            compiled_payload,
            next_source_path,
            run_config_compiler,
            _depth=_depth + 1,
        )
    return materialized


def _apply_smart_defaults(
    payload: dict[str, Any],
    source_path: Path | None,
    cli_intent: CLIIntent,
) -> dict[str, Any]:
    profile = select_smart_defaults_profile(payload, source_path)
    if not profile.rules:
        return payload
    ctx = RuleContext(source_path=source_path, cli_intent=cli_intent, scene=profile.scene)
    return apply_smart_defaults(payload, ctx, profile)
