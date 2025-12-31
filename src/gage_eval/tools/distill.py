"""Builtin template distillation helpers (Phase 1).

This module focuses on early validation for the "single-task first" policy:
- default: only a single task is allowed to be distilled;
- multi-task configs are rejected unless explicitly forced.

Later phases will add full template generation and parameter scanning on top
of these guards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from gage_eval.config.schema import SchemaValidationError, normalize_pipeline_payload


class DistillError(ValueError):
    """Raised when distillation cannot proceed."""


@dataclass(frozen=True)
class DistillTaskAnalysis:
    """Captures task layout after validation."""

    mode: str  # ATOMIC | MONOLITHIC
    task_ids: Sequence[str]
    is_monolithic: bool


def load_pipeline_payload(path: Path) -> dict:
    """Load a YAML config into a dictionary.

    This helper keeps the loader trivial; schema validation is deferred to
    :func:`analyze_tasks_for_distill`.
    """

    with path.expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise DistillError(f"Config '{path}' must be a mapping at the top level")
    return data


def analyze_tasks_for_distill(payload: dict, *, force_merge: bool = False) -> DistillTaskAnalysis:
    """Validate the payload and enforce the single-task-first policy.

    - When there is only one task (or zero explicit tasks), distillation is allowed.
    - When multiple tasks exist:
        * without ``force_merge`` -> raise DistillError with actionable guidance;
        * with ``force_merge`` -> allow and mark as monolithic.
    """

    try:
        normalized = normalize_pipeline_payload(payload)
    except SchemaValidationError as exc:
        raise DistillError(str(exc)) from exc

    tasks = normalized.get("tasks")
    tasks = tasks if isinstance(tasks, list) else []
    task_ids = [task.get("task_id") or "<unknown>" for task in tasks]
    if len(tasks) <= 1:
        return DistillTaskAnalysis(mode="ATOMIC", task_ids=tuple(task_ids), is_monolithic=False)

    if force_merge:
        return DistillTaskAnalysis(mode="MONOLITHIC", task_ids=tuple(task_ids), is_monolithic=True)

    joined = ", ".join(task_ids) if task_ids else "<unknown>"
    error = DistillError(
        "Distill rejected: multi-task config detected "
        f"({len(tasks)} tasks: {joined}). "
        "Builtin templates must stay atomic; split the config or re-run with --force-merge "
        "to create a monolithic template."
    )
    # NOTE: Attach structured context so callers can reuse the analysis result.
    error.context = DistillTaskAnalysis(mode="REJECTED", task_ids=tuple(task_ids), is_monolithic=False)  # type: ignore[attr-defined]
    raise error


def distill_to_template(
    payload: Mapping[str, object],
    *,
    builtin_name: str,
    version: str | None = None,
    output_root: Path,
    force_merge: bool = False,
) -> Path:
    """Generate a BuiltinTemplate YAML from a PipelineConfig payload.

    - Enforces single-task-first policy (unless force_merge).
    - Normalizes the payload via schema validation.
    - Computes metadata.digest over the definition block.
    """

    analysis = analyze_tasks_for_distill(payload, force_merge=force_merge)
    if analysis.mode == "REJECTED":
        raise DistillError("distill cannot proceed: tasks rejected")

    try:
        normalized = normalize_pipeline_payload(dict(payload))
    except SchemaValidationError as exc:
        raise DistillError(str(exc)) from exc

    definition = _extract_definition(normalized)
    inferred_params = _infer_parameters(definition)
    definition = _prune_empty_sections(definition, keys={"models", "parameters"})
    digest = calculate_definition_digest(definition)

    target_dir = output_root / builtin_name
    version_norm = _determine_version(version, target_dir)
    filename = f"{version_norm.lower()}.yaml"
    target = output_root / builtin_name / filename
    if target.exists():
        raise DistillError(
            f"target template already exists: {target}. "
            "Pass a new --version to bump, or remove the existing file."
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    template = {
        "api_version": "gage/v1alpha1",
        "kind": "BuiltinTemplate",
        "metadata": {
            "name": builtin_name,
            "version": version_norm,
            "digest": f"sha256:{digest}",
            "source_tasks": list(analysis.task_ids),
            "monolithic": analysis.is_monolithic,
        },
        "definition": definition,
    }
    params = normalized.get("parameters") or inferred_params
    if params:
        template["parameters"] = params
    with target.open("w", encoding="utf-8") as handle:
        handle.write(_yaml_dump(template, sort_keys=False))
    return target


def _extract_definition(normalized: Mapping[str, object]) -> dict:
    # Remove meta-level fields; keep pipeline-relevant sections.
    excluded = {"api_version", "kind"}
    definition: dict = {}
    for key, value in normalized.items():
        if key in excluded:
            continue
        definition[key] = value
    return definition


def _prune_empty_sections(mapping: Mapping[str, object], *, keys: set[str]) -> dict:
    pruned: dict = {}
    for key, value in mapping.items():
        if key in keys and (value is None or value == [] or value == {}):
            continue
        pruned[key] = value
    return pruned


def _infer_parameters(definition: Mapping[str, object]) -> list[dict]:
    params: list[dict] = []

    def _add(name: str, value, desc: str | None = None):
        if isinstance(value, bool):
            param_type = "bool"
        elif isinstance(value, int):
            param_type = "int"
        elif isinstance(value, float):
            param_type = "float"
        elif isinstance(value, dict):
            param_type = "obj"
        else:
            param_type = "str"
        entry: dict = {"name": name, "type": param_type, "default": value}
        if desc:
            entry["description"] = desc
        params.append(entry)

    for ds in definition.get("datasets") or []:
        ds_id = ds.get("dataset_id")
        if not ds_id:
            continue
        ds_copy = dict(ds)
        ds_copy.pop("dataset_id", None)
        _add(
            f"runtime.datasets.{ds_id}",
            ds_copy,
            "数据集配置块，可整体覆盖为本地/自定义来源",
        )

    for backend in definition.get("backends") or []:
        backend_id = backend.get("backend_id")
        cfg = backend.get("config") or {}
        if not backend_id:
            continue
        backend_copy = dict(backend)
        backend_copy.pop("backend_id", None)
        _add(
            f"runtime.backends.{backend_id}",
            backend_copy,
            "后端配置块，可整体覆盖为本地/自定义端点",
        )

    for task in definition.get("tasks") or []:
        task_id = task.get("task_id")
        if not task_id:
            continue
        if "max_samples" in task:
            _add(
                f"runtime.tasks.{task_id}.max_samples",
                task.get("max_samples"),
                "任务级最大样本数",
            )
        if "concurrency" in task:
            _add(
                f"runtime.tasks.{task_id}.concurrency",
                task.get("concurrency"),
                "任务调度并发",
            )
        reporting = task.get("reporting") or {}
        sinks = reporting.get("sinks") or []
        for sink in sinks:
            if sink.get("type") == "file":
                params_dict = sink.get("params") or {}
                if "output_path" in params_dict:
                    _add(
                        "runtime.global.output_path",
                        params_dict.get("output_path"),
                        "事件/结果输出路径",
                    )
                    break

    return params


def _determine_version(user_version: str | None, target_dir: Path) -> str:
    """Pick version: user-provided or auto-increment based on existing vN.yaml files."""

    def _normalize(version: str) -> str:
        v = version.upper()
        return v if v.startswith("V") else f"V{v}"

    if user_version:
        return _normalize(user_version)

    max_ver = 0
    if target_dir.exists():
        for path in target_dir.glob("v*.yaml"):
            name = path.name.lower()
            if name.startswith("v") and name.endswith(".yaml"):
                num = name[1:-5]
                if num.isdigit():
                    max_ver = max(max_ver, int(num))
    next_ver = max_ver + 1
    return f"V{next_ver}"


def calculate_definition_digest(definition: Mapping[str, object]) -> str:
    """Stable digest calculation for template definitions.

    Uses YAML dump with sort_keys=True to ensure cross-process consistency.
    Keep this in sync with any runtime digest verification logic.
    """

    payload_bytes = _yaml_dump(definition, sort_keys=True).encode("utf-8")
    from hashlib import sha256

    return sha256(payload_bytes).hexdigest()


class _LiteralDumper(yaml.SafeDumper):
    """YAML dumper that uses literal style for multi-line strings and disables anchors."""

    # NOTE: Disable YAML anchors (&id001/*id001) to keep generated templates readable.
    def ignore_aliases(self, data):
        return True


def _str_representer(dumper: yaml.SafeDumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralDumper.add_representer(str, _str_representer)


def _yaml_dump(data: Mapping[str, object], *, sort_keys: bool = False) -> str:
    return yaml.dump(
        data,
        Dumper=_LiteralDumper,
        sort_keys=sort_keys,
        allow_unicode=True,
        default_flow_style=False,
    )
