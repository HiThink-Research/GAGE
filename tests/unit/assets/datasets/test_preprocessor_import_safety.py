from __future__ import annotations

import builtins
import importlib
import sys

import pytest


@pytest.mark.fast
@pytest.mark.parametrize(
    "module_name,package_modules",
    (
        (
            "gage_eval.assets.datasets.preprocessors.biz_fin_bench_v2.biz_fin_bench_v2_converter",
            (
                "gage_eval.assets.datasets.preprocessors.biz_fin_bench_v2.biz_fin_bench_v2_converter",
                "gage_eval.assets.datasets.preprocessors.biz_fin_bench_v2",
            ),
        ),
        (
            "gage_eval.assets.datasets.preprocessors.global_piqa.global_piqa_converter",
            (
                "gage_eval.assets.datasets.preprocessors.global_piqa.global_piqa_converter",
                "gage_eval.assets.datasets.preprocessors.global_piqa",
            ),
        ),
        (
            "gage_eval.assets.datasets.preprocessors.mmsu.mmsu_converter",
            (
                "gage_eval.assets.datasets.preprocessors.mmsu.mmsu_converter",
                "gage_eval.assets.datasets.preprocessors.mmsu",
            ),
        ),
        (
            "gage_eval.assets.datasets.preprocessors.mrcr.mrcr_converter",
            (
                "gage_eval.assets.datasets.preprocessors.mrcr.mrcr_converter",
                "gage_eval.assets.datasets.preprocessors.mrcr",
            ),
        ),
        (
            "gage_eval.assets.datasets.preprocessors.mmlu_pro.mmlu_pro_converter",
            (
                "gage_eval.assets.datasets.preprocessors.mmlu_pro.mmlu_pro_converter",
                "gage_eval.assets.datasets.preprocessors.mmlu_pro",
            ),
        ),
    ),
)
def test_converter_module_import_is_transformers_safe(
    module_name: str,
    package_modules: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_modules = {name: sys.modules.pop(name, None) for name in package_modules}
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        normalized = str(name or "")
        if normalized == "transformers" or normalized.startswith("transformers."):
            raise AssertionError("transformers must be imported lazily")
        if normalized == "torch" or normalized.startswith("torch."):
            raise AssertionError("torch must not be imported during converter module import")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    try:
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name
    finally:
        for name in package_modules:
            sys.modules.pop(name, None)
        for name, previous in previous_modules.items():
            if previous is not None:
                sys.modules[name] = previous
