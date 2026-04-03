"""Terminal benchmark native verifier."""

from __future__ import annotations

import re
from typing import Any, Mapping

from gage_eval.agent_runtime.verifier.base import VerifierInput, VerifierResult
from gage_eval.sandbox.surfaces import ClientSurface

TERMINAL_BENCH_REQUIRED_SURFACES = ("terminal", "fs")


class TerminalBenchVerifier:
    """Native verifier for terminal benchmark runs."""

    def verify(self, verifier_input: VerifierInput) -> VerifierResult:
        """Verify terminal benchmark results using explicit resource and run status."""

        if verifier_input.benchmark_kit_id != "terminal_bench":
            return VerifierResult(
                status="error",
                score=0.0,
                summary=f"Unexpected benchmark_kit_id: {verifier_input.benchmark_kit_id}",
                raw_output={
                    "resolved": False,
                    "failure_reason": "unexpected_benchmark_kit_id",
                    "benchmark_kit_id": verifier_input.benchmark_kit_id,
                },
            )

        payload = verifier_input.payload if isinstance(verifier_input.payload, dict) else {}
        missing_surfaces = self._missing_required_surfaces(verifier_input, payload)
        if missing_surfaces:
            return VerifierResult(
                status="failed",
                score=0.0,
                summary=f"Missing required surfaces: {', '.join(missing_surfaces)}",
                raw_output={
                    "resolved": False,
                    "failure_reason": "missing_required_surfaces",
                    "missing_surfaces": missing_surfaces,
                    "payload": payload,
                    "runtime_handle": dict(verifier_input.runtime_handle or {}),
                    "surface_names": tuple(verifier_input.surfaces.keys()),
                },
            )

        scheduler_result = payload.get("scheduler_result") or {}
        if not isinstance(scheduler_result, dict):
            scheduler_result = {}
        if not self._scheduler_succeeded(scheduler_result):
            summary = scheduler_result.get("status") or "scheduler reported failure"
            return VerifierResult(
                status="failed",
                score=0.0,
                summary=str(summary),
                raw_output={
                    "resolved": False,
                    "failure_reason": _scheduler_failure_reason(scheduler_result),
                    "scheduler_result": scheduler_result,
                    "payload": payload,
                },
            )

        return VerifierResult(
            status="passed",
            score=1.0,
            summary="Terminal benchmark requirements satisfied.",
            raw_output={
                "resolved": True,
                "failure_reason": None,
                "scheduler_result": scheduler_result,
                "payload": payload,
                "runtime_handle": dict(verifier_input.runtime_handle or {}),
                "surface_names": tuple(verifier_input.surfaces.keys()),
                "workspace_root": verifier_input.workspace_root,
            },
        )

    @staticmethod
    def _missing_required_surfaces(
        verifier_input: VerifierInput,
        payload: Mapping[str, Any],
    ) -> tuple[str, ...]:
        actual_surfaces = {
            str(name)
            for name, surface in verifier_input.surfaces.items()
            if _surface_is_available(surface)
        }
        if not actual_surfaces:
            declared = payload.get("surface_names") or payload.get("required_surfaces") or ()
            actual_surfaces = {str(surface) for surface in declared}
        missing = tuple(surface for surface in TERMINAL_BENCH_REQUIRED_SURFACES if surface not in actual_surfaces)
        return missing

    @staticmethod
    def _scheduler_succeeded(scheduler_result: Mapping[str, Any]) -> bool:
        status = str(scheduler_result.get("status") or "").strip().lower()
        if status in {"success", "passed", "ok", "completed", "complete"}:
            return True
        if scheduler_result.get("passed") is True:
            return True
        exit_code = scheduler_result.get("exit_code")
        if exit_code is None:
            exit_code = scheduler_result.get("returncode")
        return exit_code == 0


def _surface_is_available(surface: ClientSurface) -> bool:
    return str(surface.status or "available").strip().lower() != "unavailable"


def _scheduler_failure_reason(scheduler_result: Mapping[str, Any]) -> str:
    status = str(scheduler_result.get("status") or "").strip().lower()
    if status:
        return _slugify_reason(status)
    exit_code = scheduler_result.get("exit_code")
    if exit_code is None:
        exit_code = scheduler_result.get("returncode")
    if exit_code not in (None, 0):
        return "scheduler_exit_nonzero"
    return "scheduler_failed"


def _slugify_reason(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "scheduler_failed"
