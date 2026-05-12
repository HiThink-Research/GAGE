from __future__ import annotations

import json
from pathlib import Path

import pytest

from gage_eval.external_harness_kits.harbor.trace_translation import HarborATIFTranslator


def _translate_harbor_atif(raw_trace, *, context=None):
    return HarborATIFTranslator().translate(raw_trace, context=context)


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


@pytest.mark.fast
def test_atif_v17_steps_map_to_agentkit_trace_schema() -> None:
    trace = _translate_harbor_atif(
        {
            "schema_version": "ATIF-v1.7",
            "steps": [
                {
                    "step_id": 1,
                    "timestamp": "2026-05-10T08:16:42.230838+00:00",
                    "source": "agent",
                    "message": "I will inspect the files.",
                    "tool_calls": [
                        {
                            "tool_call_id": "call-1",
                            "function_name": "bash_command",
                            "arguments": {"keystrokes": "ls -la\n", "duration": 0.5},
                        }
                    ],
                    "observation": {"results": [{"content": "New Terminal Output:\nfile.txt\n"}]},
                    "metrics": {"prompt_tokens": 10, "completion_tokens": 3, "cost_usd": 0.002},
                    "metadata": {"episode_index": 2, "latency_ms": 25.5},
                }
            ],
        }
    )

    assert len(trace) == 1
    step = trace[0]
    assert REQUIRED_TRACE_KEYS.issubset(step)
    assert step["trace_step"] == 1
    assert step["trace_role"] == "tool"
    assert step["name"] == "bash_command"
    assert step["input"] == {"keystrokes": "ls -la\n", "duration": 0.5}
    assert step["output"]["content"].startswith("New Terminal Output:")
    assert step["status"] == "success"
    assert step["latency_ms"] == 25.5
    assert step["timestamp"] == 1778401002
    assert step["turn_index"] == 2
    assert step["input_tokens"] == 10
    assert step["output_tokens"] == 3
    assert step["cost_usd"] == 0.002


@pytest.mark.fast
def test_terminus_2_fixture_tool_calls_map_to_tool_trace_role() -> None:
    fixture = Path("tests/_support/external_harness_kits/harbor_tb2_1case/result_tree.json")
    trajectory = json.loads(fixture.read_text(encoding="utf-8"))["trials"][0]["agent"]["trajectory"]

    trace = _translate_harbor_atif(trajectory)

    assert len(trace) >= len(trajectory["steps"])
    assert any(step["trace_role"] == "tool" and step["name"] == "bash_command" for step in trace)
    assert all(REQUIRED_TRACE_KEYS.issubset(step) for step in trace)


@pytest.mark.fast
def test_message_only_step_has_trace_role_assistant() -> None:
    trace = _translate_harbor_atif(
        [{"source": "agent", "timestamp": "2026-05-10T08:16:42+00:00", "message": "done"}]
    )

    assert trace == [
        {
            "trace_step": 1,
            "trace_role": "assistant",
            "name": "agent",
            "input": "done",
            "output": {"content": "done"},
            "status": "success",
            "latency_ms": 0.0,
            "timestamp": 1778401002,
        }
    ]


@pytest.mark.fast
def test_environment_observation_maps_as_previous_tool_output() -> None:
    trace = _translate_harbor_atif(
        [
            {
                "source": "agent",
                "timestamp": "2026-05-10T08:16:42+00:00",
                "tool_calls": [{"function_name": "bash", "arguments": {"cmd": "pwd"}}],
            },
            {
                "source": "environment",
                "timestamp": "2026-05-10T08:16:43+00:00",
                "message": {"content": "/app\n"},
            },
        ]
    )

    assert len(trace) == 1
    assert trace[0]["trace_role"] == "tool"
    assert trace[0]["output"] == {"content": "/app\n"}


@pytest.mark.fast
def test_error_step_status_error() -> None:
    trace = _translate_harbor_atif(
        [
            {
                "source": "agent",
                "timestamp": "2026-05-10T08:16:42+00:00",
                "tool_calls": [{"function_name": "bash", "arguments": {"cmd": "bad"}}],
                "error_info": {"message": "boom"},
            }
        ]
    )

    assert trace[0]["status"] == "error"


@pytest.mark.fast
def test_unknown_step_shape_falls_back_to_minimal_schema_with_raw_atif_v1_7_step() -> None:
    raw_step = {"timestamp": "2026-05-10T08:16:42+00:00", "unexpected": {"nested": True}}

    trace = _translate_harbor_atif([raw_step])

    assert len(trace) == 1
    assert REQUIRED_TRACE_KEYS.issubset(trace[0])
    assert trace[0]["name"] == "unknown"
    assert trace[0]["metadata"]["raw_atif_v1_7_step"] == raw_step


@pytest.mark.fast
def test_unknown_nested_message_shape_falls_back_with_raw_atif_v1_7_step() -> None:
    raw_step = {
        "source": "agent",
        "timestamp": "2026-05-10T08:16:42+00:00",
        "message": {
            "opaque_segments": [{"kind": "custom-agent-payload", "value": {"nested": True}}],
        },
    }

    trace = _translate_harbor_atif([raw_step])

    assert len(trace) == 1
    assert trace[0]["name"] == "agent"
    assert trace[0]["metadata"]["raw_atif_v1_7_step"] == raw_step


@pytest.mark.fast
def test_trajectory_level_token_usage_is_not_forced_onto_steps() -> None:
    trace = _translate_harbor_atif(
        {
            "final_metrics": {"prompt_tokens": 100, "completion_tokens": 20, "cost_usd": 0.1},
            "steps": [{"source": "agent", "message": "done"}],
        }
    )

    assert "input_tokens" not in trace[0]
    assert "output_tokens" not in trace[0]
    assert "cost_usd" not in trace[0]
