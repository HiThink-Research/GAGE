"""Config helpers."""

from __future__ import annotations

import importlib
from typing import Iterable

from gage_eval.config.loader import (
    expand_env,
    load_yaml_mapping,
    load_pipeline_config_payload,
    load_pre_smart_defaults_payload,
    materialize_pipeline_config_payload,
)
from gage_eval.config.registry import ConfigRegistry

__all__ = [
    "ConfigRegistry",
    "build_default_registry",
    "expand_env",
    "load_yaml_mapping",
    "load_pipeline_config_payload",
    "load_pre_smart_defaults_payload",
    "materialize_pipeline_config_payload",
]


def build_default_registry(plugin_modules: Iterable[str] | None = None) -> ConfigRegistry:
    """Create a ConfigRegistry and execute optional plugin hooks."""

    registry_obj = ConfigRegistry()
    for module_path in plugin_modules or ():
        module = importlib.import_module(module_path)
        register_fn = getattr(module, "register", None)
        if register_fn is None:
            raise AttributeError(
                f"Registry plugin '{module_path}' must expose a 'register(registry)' function"
            )
        register_fn(registry_obj)
    return registry_obj
