from __future__ import annotations

from pathlib import Path

from gage_eval.agent_runtime.artifacts.layout import ArtifactLayout
from gage_eval.agent_runtime.artifacts.sink import FileArtifactSink
from gage_eval.agent_runtime.artifacts.trace import TraceEvent


def test_artifact_layout_for_sample() -> None:
    layout = ArtifactLayout.for_sample("/tmp/runs", "run-1", "sample-1")

    assert layout.run_dir == "/tmp/runs/run-1"
    assert layout.sample_dir.endswith("/samples/sample-1")
    assert layout.agent_dir.endswith("/samples/sample-1/agent")
    assert layout.verifier_dir.endswith("/samples/sample-1/verifier")
    assert layout.patch_file.endswith("/samples/sample-1/agent/submission.patch")
    assert layout.metadata_file.endswith("/samples/sample-1/runtime_metadata.json")


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

