"""SWE-bench sub-workflow — shell only, not yet implemented."""

from __future__ import annotations


def prepare_inputs(sample: dict, session) -> dict:
    raise NotImplementedError("swebench.prepare_inputs is not implemented yet")


def finalize_result(sample: dict, scheduler_result, artifacts) -> dict:
    raise NotImplementedError("swebench.finalize_result is not implemented yet")
