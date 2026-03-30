"""Terminal benchmark sub-workflow — shell only, not yet implemented."""

from __future__ import annotations


def prepare_inputs(sample: dict, session) -> dict:
    raise NotImplementedError("terminal_bench.prepare_inputs is not implemented yet")


def finalize_result(sample: dict, scheduler_result, artifacts) -> dict:
    raise NotImplementedError("terminal_bench.finalize_result is not implemented yet")
