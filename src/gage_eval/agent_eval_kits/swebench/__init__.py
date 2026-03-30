"""SWE-bench benchmark kit."""

from __future__ import annotations

from gage_eval.agent_eval_kits.swebench.judge_bridge import build_verifier_input
from gage_eval.agent_eval_kits.swebench.kit import BenchmarkKitDefinition, build_kit
from gage_eval.agent_eval_kits.swebench.resources import build_resource_requirements
from gage_eval.agent_eval_kits.swebench.sub_workflow import finalize_result, prepare_inputs

__all__ = [
    "BenchmarkKitDefinition",
    "build_kit",
    "build_resource_requirements",
    "build_verifier_input",
    "finalize_result",
    "prepare_inputs",
]

