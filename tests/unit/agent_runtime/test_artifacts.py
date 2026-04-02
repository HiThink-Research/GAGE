from __future__ import annotations

from pathlib import Path

from gage_eval.agent_runtime.artifacts.layout import ArtifactLayout
from gage_eval.agent_runtime.artifacts.sink import FileArtifactSink
from gage_eval.agent_runtime.artifacts.trace import TraceEvent


def test_artifact_layout_for_sample() -> None:
    layout = ArtifactLayout.for_sample("/tmp/runs", "run-1", "sample-1")

    assert layout.run_dir == "/tmp/runs/run-1"
    assert layout.task_dir.endswith("/samples/global")
    assert layout.sample_dir.endswith("/samples/global/sample-1")
    assert layout.canonical_sample_dir.endswith("/samples/global/sample-1")
    assert layout.sample_file.endswith("/samples/global/sample-1/sample.json")
    assert layout.agent_dir.endswith("/samples/global/sample-1/agent")
    assert layout.verifier_dir.endswith("/samples/global/sample-1/verifier")
    assert layout.patch_file.endswith("/samples/global/sample-1/agent/submission.patch")
    assert layout.stderr_file.endswith("/samples/global/sample-1/agent/stderr.log")
    assert layout.final_message_file.endswith("/samples/global/sample-1/agent/final_message.md")
    assert layout.metadata_file.endswith("/samples/global/sample-1/runtime_metadata.json")
    assert layout.verifier_result_file.endswith("/samples/global/sample-1/verifier/result.json")
    assert layout.verifier_logs_dir.endswith("/samples/global/sample-1/verifier/logs")
    assert layout.verifier_workspace_dir.endswith("/samples/global/sample-1/verifier/workspace")


def test_artifact_layout_for_sample_with_task_id() -> None:
    layout = ArtifactLayout.for_sample(
        "/tmp/runs",
        "run-1",
        "sample:1/unsafe",
        task_id="task:alpha/beta",
    )

    assert layout.task_dir.endswith("/samples/task_alpha_beta")
    assert layout.sample_dir.endswith("/samples/task_alpha_beta/sample_1_unsafe")
    assert layout.canonical_sample_dir.endswith("/samples/task_alpha_beta/sample_1_unsafe")
    assert layout.sample_file.endswith("/samples/task_alpha_beta/sample_1_unsafe/sample.json")


def test_file_artifact_sink_write_text(tmp_path: Path) -> None:
    sink = FileArtifactSink()
    target = tmp_path / "nested" / "artifact.txt"

    sink.write_text(str(target), "hello")

    assert target.read_text(encoding="utf-8") == "hello"


def test_file_artifact_sink_write_bytes(tmp_path: Path) -> None:
    sink = FileArtifactSink()
    target = tmp_path / "nested" / "artifact.bin"

    sink.write_bytes(str(target), b"abc")

    assert target.read_bytes() == b"abc"


def test_trace_event_frozen() -> None:
    event = TraceEvent(name="sample", payload={"x": 1})

    assert event.name == "sample"
    assert event.payload == {"x": 1}
