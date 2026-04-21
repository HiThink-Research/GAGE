from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_swebench_smoke_imports_do_not_require_game_arena_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    src_root = repo_root / "src"
    probe = r"""
import builtins
import sys

blocked = {
    "rlcard",
    "numpy",
    "torch",
    "transformers",
    "vllm",
    "faiss",
    "librosa",
    "soundfile",
    "PIL",
    "cv2",
    "pettingzoo",
    "gymnasium",
    "vizdoom",
    "pygame",
    "modelscope",
    "anthropic",
    "litellm",
    "mcp",
}
original_import = builtins.__import__


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] in blocked:
        raise ImportError(f"blocked optional dependency: {name}")
    return original_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded_import
sys.path.insert(0, %r)

import gage_eval
import gage_eval.role.adapters
from gage_eval.config import build_default_registry
from gage_eval.role.judge.swebench_docker import SwebenchDocker
from gage_eval.assets.datasets.loaders.hf_hub_loader import HuggingFaceDatasetLoader
from gage_eval.assets.datasets.preprocessors.swebench_pro_preprocessor import SwebenchProPreprocessor
from gage_eval.agent_eval_kits.swebench import kit as swebench_kit

print("ok")
""" % str(src_root)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "ok"
