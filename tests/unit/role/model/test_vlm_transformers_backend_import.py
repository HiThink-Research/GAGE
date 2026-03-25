from __future__ import annotations

import builtins
import importlib
import sys

import pytest


@pytest.mark.fast
def test_vlm_transformers_backend_module_import_is_accelerate_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "gage_eval.role.model.backends.vlm_transformers_backend"
    previous_module = sys.modules.pop(module_name, None)
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        normalized = str(name or "")
        if normalized == "accelerate" or normalized.startswith("accelerate."):
            raise AssertionError("accelerate must be imported lazily")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    try:
        module = importlib.import_module(module_name)
        assert module.VLMTransformersBackend.__name__ == "VLMTransformersBackend"
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module
