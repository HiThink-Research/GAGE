from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gage_eval.external_harness_kits.harbor.trace_translation import HarborATIFTranslator
from gage_eval.external_harness_kits.trace_translation import fallback_minimal_step


REQUIRED_TRACE_KEYS = {
    "trace_step",
    "trace_role",
    "name",
    "input",
    "output",
    "status",
    "latency_ms",
    "timestamp",
}
ALLOWED_TRACE_ROLES = {"system", "user", "assistant", "model", "tool", "environment", "verifier"}
ALLOWED_STATUSES = {"success", "error", "aborted"}


def _load_terminus_2_trajectory() -> dict[str, Any]:
    fixture = Path("tests/_support/external_harness_kits/harbor_tb2_1case/result_tree.json")
    return json.loads(fixture.read_text(encoding="utf-8"))["trials"][0]["agent"]["trajectory"]


@pytest.mark.parametrize(
    ("translator", "raw_fixture", "expected_step_count"),
    [
        (HarborATIFTranslator(), _load_terminus_2_trajectory(), 3),
    ],
)
def test_translator_satisfies_agentkit_v2_contract(
    translator,
    raw_fixture: Any,
    expected_step_count: int,
) -> None:
    out = translator.translate(raw_fixture)

    assert isinstance(out, list)
    assert len(out) == expected_step_count
    for index, step in enumerate(out, start=1):
        assert isinstance(step, dict)
        assert REQUIRED_TRACE_KEYS.issubset(step)
        assert step["trace_step"] == index
        assert step["trace_role"] in ALLOWED_TRACE_ROLES
        assert step["status"] in ALLOWED_STATUSES


@pytest.mark.parametrize("translator", [HarborATIFTranslator()])
def test_translator_handles_empty_trace_returns_empty_list(translator) -> None:
    assert translator.translate(None) == []
    assert translator.translate([]) == []
    assert translator.translate({}) == []


@pytest.mark.parametrize("translator", [HarborATIFTranslator()])
def test_translator_handles_unknown_step_shape_falls_back_with_raw_kept(translator) -> None:
    raw_step = {
        "source": "agent",
        "timestamp": "2026-05-10T08:16:42+00:00",
        "message": {"opaque_segments": [{"kind": "custom-agent-payload"}]},
    }

    out = translator.translate([raw_step])

    assert len(out) == 1
    assert REQUIRED_TRACE_KEYS.issubset(out[0])
    assert out[0]["metadata"]["raw_atif_v1_7_step"] == raw_step


@pytest.mark.parametrize("translator", [HarborATIFTranslator()])
def test_translator_does_not_fabricate_token_usage_when_absent(translator) -> None:
    out = translator.translate({"final_metrics": {"prompt_tokens": 100}, "steps": [{"source": "agent", "message": "ok"}]})

    assert len(out) == 1
    assert "input_tokens" not in out[0]
    assert "output_tokens" not in out[0]
    assert "cost_usd" not in out[0]


def test_fallback_minimal_step_keeps_source_format_scoped_raw_step() -> None:
    raw_step = {"opaque": True}

    step = fallback_minimal_step(
        trace_step=1,
        source_format="ATIF-v1.7",
        raw_step=raw_step,
        timestamp=1778401002,
    )

    assert REQUIRED_TRACE_KEYS.issubset(step)
    assert step["trace_step"] == 1
    assert step["input"] is None
    assert step["output"] is None
    assert step["metadata"]["raw_atif_v1_7_step"] == raw_step
