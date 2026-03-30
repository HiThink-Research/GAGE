from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from gage_eval.agent_runtime.artifacts.layout import ArtifactLayout
from gage_eval.agent_runtime.artifacts.sink import FileArtifactSink
from gage_eval.agent_runtime.artifacts.trace import TraceEvent


@pytest.mark.fast
def test_artifact_layout_for_sample() -> None:
    layout = ArtifactLayout.for_sample("runs", "run-1", "sample-1")

    assert layout.sample_dir.endswith("runs/run-1/samples/sample-1")
    assert layout.patch_file.endswith("agent/submission.patch")


@pytest.mark.fast
def test_artifact_layout_paths_consistent() -> None:
    layout = ArtifactLayout.for_sample("runs", "run-1", "sample-1")

    assert layout.agent_dir.startswith(layout.sample_dir)
    assert layout.verifier_dir.startswith(layout.sample_dir)


@pytest.mark.fast
def test_file_artifact_sink_write_text(tmp_path) -> None:
    sink = FileArtifactSink()
    target = tmp_path / "artifacts" / "stdout.log"

    sink.write_text(str(target), "hello")

    assert target.read_text(encoding="utf-8") == "hello"


@pytest.mark.fast
def test_trace_event_frozen() -> None:
    event = TraceEvent(name="runtime_start", payload={"ok": True})

    with pytest.raises(FrozenInstanceError):
        event.level = "error"
