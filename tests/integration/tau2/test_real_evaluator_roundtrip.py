from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gage_eval.agent_eval_kits.tau2.artifacts import persist_tau2_artifacts
from gage_eval.agent_eval_kits.tau2.judge.scoring import (
    build_simulation,
    build_tau2_task,
    evaluate_tau2_sample,
    resolve_evaluation_type,
)

pytestmark = pytest.mark.io

pytest.importorskip("tau2")


def _sample(
    *,
    reward_basis: list[str],
    actions: list[dict] | None = None,
    communicate_info: list[str] | None = None,
    nl_assertions: list[str] | None = None,
) -> dict:
    criteria: dict[str, object] = {"reward_basis": reward_basis}
    if actions is not None:
        criteria["actions"] = actions
    if communicate_info is not None:
        criteria["communicate_info"] = communicate_info
    if nl_assertions is not None:
        criteria["nl_assertions"] = nl_assertions
    return {
        "id": "mock-real-evaluator",
        "metadata": {"tau2": {"domain": "mock", "trial": 0, "seed": 1}},
        "raw_assets": {
            "tau2": {
                "task": {
                    "id": "mock-real-evaluator",
                    "user_scenario": {"instructions": "Create a mock task."},
                    "evaluation_criteria": criteria,
                }
            }
        },
    }


def _create_task_action(**overrides: object) -> dict:
    payload: dict[str, object] = {
        "action_id": "create_important_meeting",
        "name": "create_task",
        "arguments": {"user_id": "user_1", "title": "Important Meeting"},
    }
    payload.update(overrides)
    return payload


def _create_task_messages(*, tool_content: str | None = None) -> list[dict]:
    if tool_content is None:
        tool_content = _real_create_task_tool_content()
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-create",
                    "name": "create_task",
                    "arguments": {"user_id": "user_1", "title": "Important Meeting"},
                    "requestor": "assistant",
                }
            ],
        },
        {
            "role": "tool",
            "id": "call-create",
            "content": tool_content,
            "requestor": "assistant",
        },
        {"role": "assistant", "content": "The Important Meeting task was created successfully."},
    ]


def _real_create_task_tool_content() -> str:
    from tau2.data_model.message import ToolCall
    from tau2.registry import registry

    environment = registry.get_env_constructor("mock")()
    response = environment.get_response(
        ToolCall(
            id="call-create",
            name="create_task",
            arguments={"user_id": "user_1", "title": "Important Meeting"},
            requestor="assistant",
        )
    )
    return str(response.content)


def _runtime_state(messages: list[dict]) -> dict:
    return {
        "task_id": "mock-real-evaluator",
        "domain": "mock",
        "messages": messages,
        "termination_reason": "agent_stop",
        "agent_cost": 0.0,
        "user_cost": 0.0,
    }


def test_mock_db_and_action_basis_use_real_evaluator_product_logic() -> None:
    action = _create_task_action(
        arguments={"user_id": "user_1", "title": "Important Meeting", "description": None},
        compare_args=["description"],
    )

    result = evaluate_tau2_sample(
        sample=_sample(reward_basis=["DB", "ACTION"], actions=[action]),
        runtime_state=_runtime_state(_create_task_messages()),
    )

    tau2 = result["tau2"]
    assert tau2["reward"] == 0.0
    assert tau2["reward_breakdown"] == {"DB": 1.0, "ACTION": 0.0}


def test_action_only_basis_skips_environment_replay() -> None:
    result = evaluate_tau2_sample(
        sample=_sample(reward_basis=["ACTION"], actions=[_create_task_action()]),
        runtime_state=_runtime_state(_create_task_messages(tool_content='{"unexpected": true}')),
    )

    assert result["tau2"]["reward"] == 1.0
    assert result["tau2"]["evaluation_type"].endswith("action")


def test_communicate_basis_evaluates_assistant_text() -> None:
    result = evaluate_tau2_sample(
        sample=_sample(reward_basis=["COMMUNICATE"], communicate_info=["refund approved"]),
        runtime_state=_runtime_state([{"role": "assistant", "content": "Your refund approved notice is ready."}]),
    )

    assert result["tau2"]["reward"] == 1.0
    assert result["tau2"]["reward_breakdown"] == {"COMMUNICATE": 1.0}


def test_all_with_nl_assertions_path_uses_fake_llm_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    from tau2.data_model.message import AssistantMessage
    import tau2.evaluator.evaluator_nl_assertions as nl_evaluator

    def fake_generate(**_kwargs):
        return AssistantMessage(
            role="assistant",
            content=json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": "Agent confirms completion.",
                            "reasoning": "The assistant confirmed completion.",
                            "metExpectation": True,
                        }
                    ]
                }
            ),
        )

    monkeypatch.setattr(nl_evaluator, "generate", fake_generate)

    result = evaluate_tau2_sample(
        sample=_sample(
            reward_basis=["COMMUNICATE", "NL_ASSERTION"],
            communicate_info=["task was created"],
            nl_assertions=["Agent confirms completion."],
        ),
        runtime_state=_runtime_state(
            [{"role": "assistant", "content": "The task was created and completion is confirmed."}]
        ),
    )

    assert result["tau2"]["reward"] == 1.0
    assert result["tau2"]["evaluation_type"].endswith("all_with_nl_assertions")
    assert result["tau2"]["reward_breakdown"]["NL_ASSERTION"] == 1.0


def test_runtime_state_messages_as_list_of_dicts_discriminates_message_union() -> None:
    from tau2.data_model.message import MultiToolMessage, ToolMessage

    task = build_tau2_task(_sample(reward_basis=["COMMUNICATE"], communicate_info=[]))
    simulation = build_simulation(
        task,
        _runtime_state(
            [
                {"role": "tool", "id": "call-one", "content": "ok", "requestor": "assistant"},
                {
                    "role": "tool",
                    "tool_messages": [
                        {
                            "role": "tool",
                            "id": "call-two",
                            "content": "ok",
                            "requestor": "assistant",
                        }
                    ],
                },
            ]
        ),
    )

    assert isinstance(simulation.messages[0], ToolMessage)
    assert isinstance(simulation.messages[1], MultiToolMessage)


def test_resolve_evaluation_type_selects_all_with_nl_assertions() -> None:
    task = build_tau2_task(
        _sample(
            reward_basis=["ACTION", "NL_ASSERTION"],
            actions=[_create_task_action()],
            nl_assertions=["Agent confirms completion."],
        )
    )

    evaluation_type = resolve_evaluation_type(task)
    assert getattr(evaluation_type, "value", str(evaluation_type)) == "all_with_nl_assertions"


def test_trajectory_artifact_can_reproduce_evaluator_result_offline(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample"
    artifacts_dir = sample_root / "artifacts"
    artifacts_dir.mkdir(parents=True)
    messages = [{"role": "assistant", "content": "Your refund approved notice is ready."}]

    class Runtime:
        def get_state(self) -> dict:
            return _runtime_state(messages)

    class Provider:
        def get_handle(self):
            return SimpleNamespace(sandbox=Runtime())

    persist_tau2_artifacts(
        session=SimpleNamespace(
            artifact_layout={"sample_root": str(sample_root), "artifacts_dir": str(artifacts_dir)}
        ),
        scheduler_output={},
        sandbox_provider=Provider(),
    )
    trajectory = json.loads((artifacts_dir / "tau2_trajectory.json").read_text(encoding="utf-8"))
    sample = _sample(reward_basis=["COMMUNICATE"], communicate_info=["refund approved"])

    live_result = evaluate_tau2_sample(sample=sample, runtime_state=_runtime_state(messages))
    offline_result = evaluate_tau2_sample(sample=sample, runtime_state=_runtime_state(trajectory["events"]))

    assert trajectory["source"] == "runtime_state.messages"
    assert offline_result["tau2"]["reward"] == live_result["tau2"]["reward"]
