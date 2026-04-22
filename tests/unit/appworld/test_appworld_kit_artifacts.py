from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gage_eval.agent_eval_kits.appworld.artifacts import persist_appworld_artifacts


def _build_session(tmp_path: Path) -> SimpleNamespace:
    sample_root = tmp_path / "sample"
    artifacts_dir = sample_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        artifact_layout={
            "sample_root": str(sample_root),
            "artifacts_dir": str(artifacts_dir),
        }
    )


@pytest.mark.fast
def test_persist_appworld_artifacts_exports_outputs_trace_and_logs(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    artifact_paths = persist_appworld_artifacts(
        session=session,
        scheduler_output={
            "status": "completed",
            "answer": "I found the playlist result.",
            "stdout": "tool output",
            "stderr": "",
            "agent_trace": [{"tool_name": "spotify.search"}],
        },
        saved_payload={"output": {"answer": "a love that never was"}},
    )

    assert artifact_paths == {
        "appworld_save": "artifacts/appworld_save.json",
        "appworld_outputs": "artifacts/appworld_outputs.json",
        "appworld_tool_trace": "artifacts/appworld_tool_trace.json",
        "appworld_logs": "artifacts/appworld_logs.json",
    }
    outputs_payload = json.loads((tmp_path / "sample" / "artifacts" / "appworld_outputs.json").read_text())
    tool_trace_payload = json.loads((tmp_path / "sample" / "artifacts" / "appworld_tool_trace.json").read_text())
    logs_payload = json.loads((tmp_path / "sample" / "artifacts" / "appworld_logs.json").read_text())

    assert outputs_payload == {
        "output": {"answer": "a love that never was"},
        "is_empty": False,
    }
    assert tool_trace_payload == {"agent_trace": [{"tool_name": "spotify.search"}]}
    assert logs_payload["save_output_is_empty"] is False
    assert logs_payload["status"] == "completed"
    assert logs_payload["tool_trace_step_count"] == 1
    assert logs_payload["tool_trace_is_empty"] is False


@pytest.mark.fast
def test_persist_appworld_artifacts_marks_empty_save_output(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    persist_appworld_artifacts(
        session=session,
        scheduler_output={"status": "failed", "answer": "", "agent_trace": []},
        saved_payload={"output": None},
    )

    outputs_payload = json.loads((tmp_path / "sample" / "artifacts" / "appworld_outputs.json").read_text())
    logs_payload = json.loads((tmp_path / "sample" / "artifacts" / "appworld_logs.json").read_text())

    assert outputs_payload == {"output": None, "is_empty": True}
    assert logs_payload["save_payload_has_output_key"] is True
    assert logs_payload["save_output_is_empty"] is True
    assert logs_payload["tool_trace_is_empty"] is True
