from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gage_eval.agent_eval_kits.tau2.artifacts import persist_tau2_artifacts
from gage_eval.agent_eval_kits.tau2.config_schema import (
    Tau2KitConfig,
    normalize_user_simulator_config,
)
from gage_eval.agent_eval_kits.tau2.sub_workflows.framework_loop import _finalize_loop_result
from gage_eval.agent_eval_kits.tau2.local_runtime import (
    CANONICAL,
    Tau2Runtime,
    TerminalSignal,
    _normalize_user_message_tool_calls,
    _normalize_user_tool_name,
    parse_terminal_signal,
)
from tests._support.stubs.tau2_stub import install_tau2_stub


def _sample(domain: str = "airline") -> dict:
    return {
        "id": "tau2-sample-1",
        "metadata": {"tau2": {"domain": domain, "trial": 0, "seed": 1}},
        "raw_assets": {
            "tau2": {
                "task": {
                    "id": "task-1",
                    "user_scenario": {"instructions": "Need support"},
                    "evaluation_criteria": {"reward_basis": ["DB"]},
                }
            }
        },
    }


class _RuntimeForArtifacts:
    def __init__(self, state: dict[str, object]) -> None:
        self.state = state
        self.mark_calls: list[str] = []
        self.recorded_usage: object = None

    def get_state(self) -> dict[str, object]:
        return dict(self.state)

    def mark_agent_exhausted(self, detail: str) -> None:
        self.mark_calls.append(detail)
        if self.state.get("termination_reason") is None:
            self.state["termination_reason"] = "agent_error"
            self.state["termination_detail"] = detail

    def record_agent_usage(self, usage: object) -> None:
        self.recorded_usage = usage
        if isinstance(usage, dict):
            self.state["agent_cost"] = usage.get("cost_usd")
            self.state["agent_total_tokens"] = usage.get("total_tokens")


class _Provider:
    def __init__(self, runtime: object) -> None:
        self.runtime = runtime

    def get_handle(self) -> SimpleNamespace:
        return SimpleNamespace(sandbox=self.runtime)


def _session(tmp_path: Path) -> SimpleNamespace:
    sample_root = tmp_path / "sample"
    artifacts_dir = sample_root / "artifacts"
    artifacts_dir.mkdir(parents=True)
    return SimpleNamespace(
        artifact_layout={
            "sample_root": str(sample_root),
            "artifacts_dir": str(artifacts_dir),
        }
    )


def test_tau2_synthetic_prefix_len_not_visible_to_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"runtime_configs": {"data_dir": str(tmp_path)}})
    sample = _sample()

    init_output = runtime.initialize_task(sample)

    assert [message["role"] for message in init_output["messages"]] == ["user"]
    assert "Hi! How can I help you today?" not in json.dumps(init_output["messages"])
    assert len(runtime.get_state()["messages"]) == 2


def test_tau2_parse_terminal_signal_variants() -> None:
    assert parse_terminal_signal("### out of scope ###") == TerminalSignal(
        kind="OUT_OF_SCOPE",
        canonical=CANONICAL["OUT_OF_SCOPE"],
        raw="### out of scope ###",
    )
    assert parse_terminal_signal("Thanks.\n### STOP ###") == TerminalSignal(
        kind="STOP",
        canonical=CANONICAL["STOP"],
        raw="### STOP ###",
    )
    assert parse_terminal_signal("### transfer ###").canonical == CANONICAL["TRANSFER"]
    assert parse_terminal_signal("### transfer ###", allowed=frozenset({"STOP"})) is None


def test_tau2_normalize_user_message_tool_calls() -> None:
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(name="check_roaming_status<|channel|>analysis"),
            SimpleNamespace(name="get_status_bar"),
        ]
    )

    normalized = _normalize_user_message_tool_calls(message)

    assert [call.name for call in normalized.tool_calls] == [
        "check_network_status",
        "check_status_bar",
    ]
    assert [call.name for call in message.tool_calls] == [
        "check_roaming_status<|channel|>analysis",
        "get_status_bar",
    ]


def test_tau2_tool_namespace_helpers_match_tau2patch() -> None:
    assert _normalize_user_tool_name("check_data_saver_mode") == "check_data_restriction_status"
    assert _normalize_user_tool_name("get_status_bar<|channel|>commentary") == "check_status_bar"
    assert _normalize_user_tool_name("run_speed_test") == "run_speed_test"


def test_tau2_openai_http_user_simulator_config_normalization() -> None:
    normalized = normalize_user_simulator_config(
        {
            "type": "openai_http",
            "model": "qwen3-8b",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
            "temperature": 0.2,
        }
    )

    assert normalized == {
        "model": "qwen3-8b",
        "model_args": {
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
            "temperature": 0.2,
        },
    }
    assert Tau2KitConfig.model_validate({"domain": "telecom", "user_simulator": normalized}).user_simulator == normalized


@pytest.mark.parametrize(
    ("loop_exit_reason", "expected_detail"),
    [
        ("tool_call_retry_budget", "no_tool_call_from_agent"),
        ("max_turns", "agent_loop_max_turns"),
    ],
)
def test_tau2_mark_loop_termination_writes_agent_exhausted_and_terminal_detail(
    tmp_path: Path,
    loop_exit_reason: str,
    expected_detail: str,
) -> None:
    runtime = _RuntimeForArtifacts(
        {
            "messages": [],
            "termination_reason": None,
            "termination_detail": None,
            "agent_cost": 0.0,
            "user_cost": 0.0,
        }
    )

    output = _finalize_loop_result(
        session=_session(tmp_path),
        sample={},
        scheduler_output={"loop_exit_reason": loop_exit_reason},
        sandbox_provider=_Provider(runtime),
    )

    assert runtime.mark_calls == [expected_detail]
    assert output["runtime_state"]["termination_detail"] == expected_detail
    state_payload = json.loads((tmp_path / "sample" / "artifacts" / "tau2_state.json").read_text())
    assert state_payload["agent_exhausted"] is True
    assert state_payload["termination_detail"] == expected_detail
    assert output["artifact_paths"]["tau2_cost"] == "artifacts/tau2_cost.json"


def test_tau2_trajectory_source_prefers_runtime_messages_over_agent_trace(tmp_path: Path) -> None:
    runtime = _RuntimeForArtifacts(
        {
            "messages": [{"role": "assistant", "content": "runtime message"}],
            "termination_reason": "agent_stop",
            "agent_cost": 0.0,
            "user_cost": 0.0,
        }
    )

    persist_tau2_artifacts(
        session=_session(tmp_path),
        scheduler_output={"agent_trace": [{"name": "stale_trace"}]},
        sandbox_provider=_Provider(runtime),
    )

    trajectory = json.loads((tmp_path / "sample" / "artifacts" / "tau2_trajectory.json").read_text())
    assert trajectory["source"] == "runtime_state.messages"
    assert trajectory["events"] == [{"role": "assistant", "content": "runtime message"}]


def test_tau2_record_agent_usage_preserved_in_summary(tmp_path: Path) -> None:
    runtime = _RuntimeForArtifacts(
        {
            "messages": [],
            "termination_reason": "agent_stop",
            "agent_cost": None,
            "user_cost": 0.0,
        }
    )

    _finalize_loop_result(
        session=_session(tmp_path),
        sample={},
        scheduler_output={"usage": {"total_tokens": 42, "cost_usd": 0.125}},
        sandbox_provider=_Provider(runtime),
    )

    assert runtime.recorded_usage == {"total_tokens": 42, "cost_usd": 0.125}
    state_payload = json.loads((tmp_path / "sample" / "artifacts" / "tau2_state.json").read_text())
    cost_payload = json.loads((tmp_path / "sample" / "artifacts" / "tau2_cost.json").read_text())
    assert state_payload["agent_cost"] == 0.125
    assert state_payload["agent_total_tokens"] == 42.0
    assert cost_payload["agent_cost"] == 0.125
    assert cost_payload["agent_total_tokens"] == 42.0
