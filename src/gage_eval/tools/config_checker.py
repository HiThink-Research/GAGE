"""Utility CLI that validates PipelineConfig YAML files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import yaml

from gage_eval.config.loader import load_pipeline_config_payload, materialize_pipeline_config_payload
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.evaluation.task_plan import build_task_plan_specs
from gage_eval.tools.distill import calculate_definition_digest

_DEPRECATED_DATASET_PREPROCESSORS = {
    "piqa_struct_only": (
        "global_piqa_chat_preprocessor",
        "config/custom/global_piqa/global_piqa_chat.yaml",
    ),
    "gpqa_multi_choice": (
        "gpqa_diamond_multi_choice",
        "config/custom/gpqa_diamond/async_chat.yaml",
    ),
    "gpqa_struct_only": (
        "gpqa_diamond_multi_choice",
        "config/custom/gpqa_diamond/async_chat.yaml",
    ),
    "mathvista_preprocessor": (
        "mathvista_chat_preprocessor",
        "config/custom/mathvista/chat.yaml",
    ),
    "mathvista_struct_only": (
        "mathvista_chat_preprocessor",
        "config/custom/mathvista/chat.yaml",
    ),
}

_NONSTANDARD_LITELLM_PROVIDER_ALIASES = {
    "openai_compatible": "openai",
}

_KNOWN_LITELLM_MODEL_PREFIXES = {
    "openai",
    "azure",
    "anthropic",
    "bedrock",
    "cohere",
    "deepseek",
    "gemini",
    "groq",
    "hosted_vllm",
    "lm_studio",
    "mistral",
    "ollama",
    "ollama_chat",
    "openrouter",
    "together_ai",
    "vertex_ai",
    "xai",
}


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

    # NOTE: Validate digest (if present) against the definition block to catch
    # "definition changed but digest not updated" issues early.
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
    pipeline_payload = materialize_pipeline_config_payload(pipeline_payload, source_path=None)
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


def lint_pipeline_config_payload(payload: dict) -> list[str]:
    """Return static config lint issues that schema validation cannot catch."""

    issues: list[str] = []
    for index, dataset in enumerate(payload.get("datasets") or []):
        if not isinstance(dataset, dict):
            continue
        params = dataset.get("params") if isinstance(dataset.get("params"), dict) else {}
        preprocessor = params.get("preprocess")
        if isinstance(preprocessor, str) and preprocessor in _DEPRECATED_DATASET_PREPROCESSORS:
            replacement, replacement_config = _DEPRECATED_DATASET_PREPROCESSORS[preprocessor]
            dataset_id = dataset.get("dataset_id") or f"#{index}"
            issues.append(
                "datasets[{index}] {dataset_id}: params.preprocess '{preprocessor}' is deprecated; "
                "use '{replacement}' via '{replacement_config}' instead.".format(
                    index=index,
                    dataset_id=dataset_id,
                    preprocessor=preprocessor,
                    replacement=replacement,
                    replacement_config=replacement_config,
                )
            )

    for index, backend in enumerate(payload.get("backends") or []):
        if not isinstance(backend, dict) or backend.get("type") != "litellm":
            continue
        config = backend.get("config") if isinstance(backend.get("config"), dict) else {}
        backend_id = backend.get("backend_id") or f"#{index}"
        provider = _string_value(config.get("provider"))
        custom_provider = _string_value(config.get("custom_llm_provider"))
        for key, value in (("provider", provider), ("custom_llm_provider", custom_provider)):
            if value in _NONSTANDARD_LITELLM_PROVIDER_ALIASES:
                issues.append(
                    "backends[{index}] {backend_id}: config.{key} '{value}' is not a LiteLLM provider; "
                    "use '{replacement}' and keep OpenAI-compatible endpoints in config.api_base.".format(
                        index=index,
                        backend_id=backend_id,
                        key=key,
                        value=value,
                        replacement=_NONSTANDARD_LITELLM_PROVIDER_ALIASES[value],
                    )
                )
        model = _string_value(config.get("model"))
        if not model:
            continue
        if _has_known_litellm_model_prefix(model) or provider or custom_provider:
            continue
        issues.append(
            "backends[{index}] {backend_id}: litellm config.model '{model}' has no known provider prefix; "
            "set config.provider or config.custom_llm_provider explicitly.".format(
                index=index,
                backend_id=backend_id,
                model=model,
            )
        )
    return issues


def _string_value(value: object) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _has_known_litellm_model_prefix(model: str) -> bool:
    prefix = model.split("/", 1)[0].strip().lower()
    return "/" in model and prefix in _KNOWN_LITELLM_MODEL_PREFIXES


def validate_config(path: Path, *, materialize_runtime: bool = False) -> None:
    payload = _load_yaml(path)
    kind = (payload.get("kind") or "").lower()
    if kind in {"builtintemplate", "builtin"}:
        _validate_builtin_template(payload, materialize_runtime=materialize_runtime)
        print(f"[gage-eval] ✓ (BuiltinTemplate) {path}")
        return

    payload = load_pipeline_config_payload(path)
    lint_issues = lint_pipeline_config_payload(payload)
    if lint_issues:
        rendered = "\n".join(f"- {issue}" for issue in lint_issues)
        raise ValueError(f"Config lint failed for '{path}':\n{rendered}")
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
