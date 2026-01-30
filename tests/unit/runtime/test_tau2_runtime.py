from __future__ import annotations

from pathlib import Path

from gage_eval.sandbox.tau2_runtime import Tau2Runtime
from tests.tau2_stub import install_tau2_stub, STOP


def _build_sample(domain: str = "airline") -> dict:
    task = {
        "id": "task-1",
        "user_scenario": {"instructions": "Call support"},
        "evaluation_criteria": {"reward_basis": ["DB"]},
    }
    return {
        "id": "sample-1",
        "metadata": {"tau2": {"domain": domain, "trial": 0, "seed": 1}},
        "raw_assets": {"tau2": {"task": task}},
    }


def test_tau2_runtime_basic_flow(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"runtime_configs": {"data_dir": str(tmp_path)}})

    sample = _build_sample()
    init_output = runtime.initialize_task(sample)

    assert init_output["messages"]
    assert len(sample["messages"]) == 2

    respond_out = runtime.exec_tool("respond", {"message": "hello"})
    assert respond_out["user_message"] == "user_response"

    tool_out = runtime.exec_tool("lookup", {"query": "x"})
    assert "content" in tool_out


def test_tau2_runtime_user_tools_and_stop(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path, force_user_tool_call=True)
    runtime = Tau2Runtime()
    runtime.start({"runtime_configs": {"data_dir": str(tmp_path)}})

    sample = _build_sample(domain="telecom")
    runtime.initialize_task(sample)
    respond_out = runtime.exec_tool("respond", {"message": "please stop"})

    assert respond_out["final_answer"] == STOP
    state = runtime.get_state()
    term = state["termination_reason"]
    assert (term.value if hasattr(term, "value") else str(term)) == "user_stop"
