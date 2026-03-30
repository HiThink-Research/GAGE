from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import run as gage_run


@pytest.mark.fast
def test_detect_gpu_count_skips_torch_auto_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        normalized = str(name or "")
        if normalized == "torch" or normalized.startswith("torch."):
            raise AssertionError("_detect_gpu_count must not auto-import torch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)

    count = gage_run._detect_gpu_count()

    assert count is None or isinstance(count, int)
