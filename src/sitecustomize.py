"""Interpreter bootstrap hooks for spawned worker processes."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _install_vllm_renderer_patch() -> None:
    """Load the lightweight vLLM renderer patch module without importing `gage_eval`."""

    module_path = Path(__file__).resolve().parent / "gage_eval" / "compat" / "vllm_renderer_patch.py"
    if not module_path.exists():
        return

    spec = importlib.util.spec_from_file_location("_gage_vllm_renderer_patch_site", module_path)
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return

    install = getattr(module, "install_vllm_renderer_compat_patches", None)
    if callable(install):
        try:
            install()
        except Exception:
            return


_install_vllm_renderer_patch()
