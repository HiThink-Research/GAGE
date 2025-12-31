"""Load registry assets on package import."""

from __future__ import annotations

import warnings

from gage_eval._loguru_compat import ensure_loguru
from gage_eval.registry import registry

ensure_loguru()

_AUTO_DISCOVERY_PACKAGES = {
    "backends": ("gage_eval.role.model.backends",),
    "roles": ("gage_eval.role.adapters", "gage_eval.role.model"),
    "context_impls": ("gage_eval.role.context",),
    "judge_impls": ("gage_eval.role.judge",),
    "arena_impls": ("gage_eval.role.arena.games",),
    "parser_impls": ("gage_eval.role.arena.parsers",),
    "renderer_impls": ("gage_eval.role.arena.games",),
    "dataset_hubs": ("gage_eval.assets.datasets.hubs",),
    "dataset_loaders": ("gage_eval.assets.datasets.loaders",),
    "bundles": ("gage_eval.assets.datasets.bundles.builtin",),
    "dataset_preprocessors": ("gage_eval.assets.datasets.preprocessors.builtin",),
    "metrics": ("gage_eval.metrics.builtin",),
    "prompts": ("gage_eval.assets.prompts.catalog",),
    "model_hubs": ("gage_eval.assets.models.hubs",),
    "observability_plugins": ("gage_eval.observability.plugins",),
    "pipeline_steps": ("gage_eval.pipeline.steps",),
}

for kind, packages in _AUTO_DISCOVERY_PACKAGES.items():
    for package in packages:
        try:
            registry.auto_discover(kind, package)
        except Exception as exc:  # pragma: no cover - defensive logging
            warnings.warn(f"Failed to auto-discover '{kind}' assets from {package}: {exc}", RuntimeWarning)
