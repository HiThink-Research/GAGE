"""Tau2 sub-workflow — shell only, not yet implemented."""

from __future__ import annotations


def prepare_inputs(sample: dict, session) -> dict:
    raise NotImplementedError("tau2.prepare_inputs is not implemented yet")


def finalize_result(sample: dict, scheduler_result, artifacts) -> dict:
    raise NotImplementedError("tau2.finalize_result is not implemented yet")
