from __future__ import annotations

import pytest

from gage_eval.config.pipeline_config import CustomPipelineStep
from gage_eval.evaluation.support_artifacts import (
    record_support_output,
    resolve_support_field,
    resolve_support_filters,
    resolve_support_tools,
)
from gage_eval.evaluation.task_planner import TaskPlanner


@pytest.mark.fast
def test_task_planner_assigns_stable_support_slot_ids() -> None:
    planner = TaskPlanner()
    planner.configure_custom_steps(
        (
            CustomPipelineStep(step_type="support", adapter_id="api_docs"),
            CustomPipelineStep(step_type="support", adapter_id="toolchain_main"),
            CustomPipelineStep(step_type="inference", adapter_id="dut"),
        )
    )

    plan = planner.prepare_plan(sample={"id": "sample-1"})

    assert [step.params["support_slot_id"] for step in plan.support_steps] == [
        "support:00:api_docs",
        "support:01:toolchain_main",
    ]


@pytest.mark.fast
def test_support_artifacts_compact_outputs_and_merge_tools() -> None:
    sample = {
        "tools": [
            {
                "name": "base__ping",
                "description": "Base ping",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
    }
    record_support_output(
        sample,
        slot_id="support:00:api_docs",
        adapter_id="api_docs",
        output={
            "tool_documentation": "DOCS-V1",
            "tool_allowlist": ["spotify__login"],
            "tools_schema": [
                {
                    "name": "spotify__login",
                    "description": "Old login",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
    )
    record_support_output(
        sample,
        slot_id="support:01:toolchain_main",
        adapter_id="toolchain_main",
        output={
            "tool_prefixes": ["supervisor"],
            "tool_doc_allowed_apps": ["spotify"],
            "tool_max_tools": 5,
            "answer": "cleaned",
            "tools_schema": [
                {
                    "name": "spotify__login",
                    "description": "New login",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "name": "supervisor__complete_task",
                    "description": "Complete task",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
        },
    )

    artifacts = sample["support_artifacts"]
    names = [tool["function"]["name"] for tool in resolve_support_tools(sample)]

    assert artifacts["stats"]["entry_count"] == 2
    assert sample["support_outputs"][0]["slot_id"] == "support:00:api_docs"
    assert resolve_support_field(sample, "tool_documentation") == "DOCS-V1"
    assert resolve_support_field(sample, "answer") == "cleaned"
    assert names == ["base__ping", "spotify__login", "supervisor__complete_task"]
    assert resolve_support_tools(sample)[1]["function"]["description"] == "New login"
    assert resolve_support_filters(sample) == {
        "allowlist": ["spotify__login"],
        "prefixes": ["supervisor"],
        "doc_allowed_apps": ["spotify"],
        "max_tools": 5,
    }
