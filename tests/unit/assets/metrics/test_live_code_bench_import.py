from __future__ import annotations

import builtins
import importlib
import sys

import pytest


@pytest.mark.fast
def test_live_code_bench_metric_import_is_datasets_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "gage_eval.metrics.builtin.live_code_bench.pass_k"
    package_modules = (
        module_name,
        "gage_eval.assets.datasets.loaders.live_code_bench.code_generation",
        "gage_eval.assets.datasets.loaders.live_code_bench.loader",
        "gage_eval.assets.datasets.loaders.live_code_bench",
    )
    previous_modules = {name: sys.modules.pop(name, None) for name in package_modules}
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        normalized = str(name or "")
        if normalized == "datasets" or normalized.startswith("datasets."):
            raise AssertionError("datasets must be imported lazily")
        if normalized == "torch" or normalized.startswith("torch."):
            raise AssertionError("torch must not be imported during metric module import")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    try:
        module = importlib.import_module(module_name)
        assert module.LiveCodeBenchPassMetric.__name__ == "LiveCodeBenchPassMetric"
    finally:
        for name in package_modules:
            sys.modules.pop(name, None)
        for name, previous in previous_modules.items():
            if previous is not None:
                sys.modules[name] = previous
