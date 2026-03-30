"""Terminal benchmark kit package."""

from __future__ import annotations

from gage_eval.agent_eval_kits.terminal_bench.contracts import (
    BenchmarkKitDefinition,
    TERMINAL_BENCH_DEFAULT_TIMEOUT_SEC,
    TERMINAL_BENCH_KIT_ID,
    TERMINAL_BENCH_REQUIRED_SURFACES,
    TerminalBenchTaskContext,
)
from gage_eval.agent_eval_kits.terminal_bench.judge_bridge import build_verifier_input
from gage_eval.agent_eval_kits.terminal_bench.kit import build_kit
from gage_eval.agent_eval_kits.terminal_bench.resources import build_resource_requirements
from gage_eval.agent_eval_kits.terminal_bench.sub_workflow import finalize_result, prepare_inputs
from gage_eval.agent_eval_kits.terminal_bench.trace_mapping import normalize_trace_events
from gage_eval.agent_eval_kits.terminal_bench.units import build_task_context, get_instruction, get_sample_id

__all__ = [
    "BenchmarkKitDefinition",
    "TERMINAL_BENCH_DEFAULT_TIMEOUT_SEC",
    "TERMINAL_BENCH_KIT_ID",
    "TERMINAL_BENCH_REQUIRED_SURFACES",
    "TerminalBenchTaskContext",
    "build_kit",
    "build_resource_requirements",
    "build_task_context",
    "build_verifier_input",
    "finalize_result",
    "get_instruction",
    "get_sample_id",
    "normalize_trace_events",
    "prepare_inputs",
]
