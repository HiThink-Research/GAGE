from __future__ import annotations

import pytest

from gage_eval.agent_runtime.verifier.base import VerifierInput
from gage_eval.agent_runtime.verifier.terminal_bench import TerminalBenchVerifier
from gage_eval.sandbox.surfaces import ClientSurface


@pytest.mark.fast
def test_terminal_bench_verifier_passes_with_required_surfaces() -> None:
    verifier = TerminalBenchVerifier()
    verifier_input = VerifierInput(
        benchmark_kit_id="terminal_bench",
        sample_id="tb2__smoke_1",
        payload={
            "scheduler_result": {"status": "success"},
        },
        surfaces={
            "terminal": ClientSurface(surface_type="terminal"),
            "fs": ClientSurface(surface_type="fs"),
        },
    )

    result = verifier.verify(verifier_input)

    assert result.status == "passed"
    assert result.score == 1.0
    assert result.raw_output["resolved"] is True
    assert result.raw_output["failure_reason"] is None


@pytest.mark.fast
def test_terminal_bench_verifier_fails_without_terminal_surface() -> None:
    verifier = TerminalBenchVerifier()
    verifier_input = VerifierInput(
        benchmark_kit_id="terminal_bench",
        sample_id="tb2__smoke_1",
        payload={
            "surface_names": ("fs",),
            "scheduler_result": {"status": "success"},
        },
    )

    result = verifier.verify(verifier_input)

    assert result.status == "failed"
    assert result.score == 0.0
    assert result.raw_output["resolved"] is False
    assert result.raw_output["failure_reason"] == "missing_required_surfaces"


@pytest.mark.fast
def test_terminal_bench_verifier_normalizes_scheduler_failure_reason() -> None:
    verifier = TerminalBenchVerifier()
    verifier_input = VerifierInput(
        benchmark_kit_id="terminal_bench",
        sample_id="tb2__smoke_1",
        payload={
            "scheduler_result": {"status": "Runtime Error", "exit_code": 1},
        },
        surfaces={
            "terminal": ClientSurface(surface_type="terminal"),
            "fs": ClientSurface(surface_type="fs"),
        },
    )

    result = verifier.verify(verifier_input)

    assert result.status == "failed"
    assert result.raw_output["resolved"] is False
    assert result.raw_output["failure_reason"] == "runtime_error"
