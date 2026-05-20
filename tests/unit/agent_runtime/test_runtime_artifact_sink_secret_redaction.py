from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink


@pytest.mark.fast
def test_runtime_artifact_sink_redacts_runtime_metadata_and_raw_error(tmp_path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    layout = sink.build_layout(run_id="run", task_id="task", sample_id="sample")
    session = SimpleNamespace(
        artifact_layout=layout,
        session_id="session",
        run_id="run",
        task_id="task",
        sample_id="sample",
        benchmark_kit_id="kit",
        scheduler_type="scheduler",
        client_id="client",
        resource_lease=None,
        runtime_context={"Authorization": "Bearer abc123"},
        prompt_context={"email": "user@example.com"},
        benchmark_state={},
        scheduler_state={"password": "secret"},
    )

    metadata_path = sink.persist_runtime_metadata(session=session)
    error_path = sink.persist_raw_error(session=session, error=RuntimeError("sk-abcdefghijklmnopqrstuvwxyz1234567890"))

    serialized = json.dumps(
        [
            json.loads(open(metadata_path, encoding="utf-8").read()),
            json.loads(open(error_path, encoding="utf-8").read()),
        ],
        ensure_ascii=False,
    )
    assert "Bearer abc123" not in serialized
    assert "user@example.com" not in serialized
    assert "secret" not in serialized
    assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in serialized
    assert "<redacted:" in serialized


@pytest.mark.fast
def test_runtime_artifact_sink_redacts_trace_event_json_arguments(tmp_path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))

    ref = sink.append_trace_event(
        run_id="run",
        task_id="task",
        sample_id="sample",
        trial_id="trial_0001",
        actor="agent",
        event_type="tool.call.raw",
        payload={
            "raw_message": {
                "function": {
                    "arguments": '{"username":"user@example.com","password":"password123"}'
                }
            }
        },
    )

    text = (tmp_path / "run" / ref.path).read_text(encoding="utf-8")
    assert "user@example.com" not in text
    assert "password123" not in text
    assert "<redacted:email>" in text
    assert "<redacted:credential>" in text
