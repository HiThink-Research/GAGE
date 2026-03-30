from __future__ import annotations

import builtins

import pytest

from gage_eval.role.model.backends.vllm.vllm_request import (
    detect_vllm_version,
    ensure_spawn_start_method,
    resolve_sampling_class,
)


@pytest.mark.fast
def test_ensure_spawn_start_method_is_torch_import_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        normalized = str(name or "")
        if normalized == "torch" or normalized.startswith("torch."):
            raise AssertionError("ensure_spawn_start_method must not import torch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)

    ensure_spawn_start_method()

    assert True


@pytest.mark.fast
def test_detect_vllm_version_is_import_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        normalized = str(name or "")
        if normalized == "vllm" or normalized.startswith("vllm."):
            raise AssertionError("detect_vllm_version must not import vllm")
        if normalized == "torch" or normalized.startswith("torch."):
            raise AssertionError("detect_vllm_version must not import torch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    version = detect_vllm_version()

    assert version is None or isinstance(version, str)


@pytest.mark.fast
def test_resolve_sampling_class_is_import_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        normalized = str(name or "")
        if normalized == "vllm" or normalized.startswith("vllm."):
            raise AssertionError("resolve_sampling_class must not import vllm")
        if normalized == "torch" or normalized.startswith("torch."):
            raise AssertionError("resolve_sampling_class must not import torch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    sampling_class = resolve_sampling_class()
    instance = sampling_class(temperature=0.0)

    assert getattr(instance, "temperature") == 0.0
