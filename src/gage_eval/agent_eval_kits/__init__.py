from __future__ import annotations

import importlib

from gage_eval.agent_eval_kits.common import BenchmarkKitEntry


_KIT_MODULES = {
    "appworld": "gage_eval.agent_eval_kits.appworld.kit",
    "swebench": "gage_eval.agent_eval_kits.swebench.kit",
    "skillsbench": "gage_eval.agent_eval_kits.swebench.kit",
    "tau2": "gage_eval.agent_eval_kits.tau2.kit",
    "terminal_bench": "gage_eval.agent_eval_kits.terminal_bench.kit",
}


def load_benchmark_kit(benchmark_kit_id: str) -> BenchmarkKitEntry:
    """Load one benchmark kit entry by id."""

    module_path = _KIT_MODULES.get(benchmark_kit_id)
    if module_path is None:
        raise KeyError(f"Unknown benchmark kit '{benchmark_kit_id}'")
    module = importlib.import_module(module_path)
    return module.load_kit()


__all__ = [
    "BenchmarkKitEntry",
    "load_benchmark_kit",
]
