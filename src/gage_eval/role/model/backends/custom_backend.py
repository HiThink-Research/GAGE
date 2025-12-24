"""Backend that loads user-defined EngineBackend implementations from scripts."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Dict, Type

from gage_eval.registry import registry
from gage_eval.role.model.backends.base_backend import EngineBackend


@registry.asset(
    "backends",
    "custom_script",
    desc="Load a custom EngineBackend from an external Python file",
    tags=("custom",),
)
class CustomScriptBackend(EngineBackend):
    def __init__(self, config: Dict[str, Any]) -> None:
        self._module_path = config.get("module_path")
        self._file_path = config.get("file_path")
        self._class_name = config.get("class_name")
        if not (self._module_path or self._file_path):
            raise ValueError("custom_script backend requires 'module_path' or 'file_path'")
        super().__init__(config)

    def load_model(self, config: Dict[str, Any]):
        module = self._load_module()
        backend_cls = self._locate_backend_cls(module)
        return backend_cls(config)

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("CustomScriptBackend should delegate to the loaded backend instance")

    def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.model.ainvoke(payload)

    def _load_module(self):
        if self._module_path:
            return importlib.import_module(self._module_path)
        spec = importlib.util.spec_from_file_location("custom_backend_module", Path(self._file_path).resolve())
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from file '{self._file_path}'")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _locate_backend_cls(self, module) -> Type[EngineBackend]:
        if self._class_name:
            candidate = getattr(module, self._class_name, None)
            if candidate and inspect.isclass(candidate) and issubclass(candidate, EngineBackend):
                return candidate
            raise ValueError(f"Class '{self._class_name}' not found or not an EngineBackend")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, EngineBackend) and obj is not EngineBackend:
                return obj
        raise ValueError("No EngineBackend subclass found in custom module")
