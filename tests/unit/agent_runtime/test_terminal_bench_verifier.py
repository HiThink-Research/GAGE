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
