import base64
import os
from pathlib import Path
from typing import Callable, List

import pytest

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


@pytest.fixture
def test_data_dir() -> Path:
    """Return absolute path to tests/data."""

    return Path(__file__).resolve().parent / "data"


@pytest.fixture
def temp_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set GAGE_EVAL_SAVE_DIR to a temp dir and restore env after."""

    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def mock_trace() -> ObservabilityTrace:
    """In-memory trace for assertions."""

    return ObservabilityTrace(recorder=InMemoryRecorder(run_id="test-trace"))


@pytest.fixture
def sample_dataset_factory() -> Callable[[int, str], List[dict]]:
    """Generate sample datasets with optional modality."""

    def _make(count: int = 1, modality: str = "text") -> List[dict]:
        samples = []
        for idx in range(count):
            content = [{"type": "text", "text": "hi"}]
            if modality == "image":
                content.append({"type": "image_url", "image_url": {"url": f"img{idx}.png"}})
            samples.append(
                {
                    "id": f"s{idx}",
                    "messages": [{"role": "user", "content": content}],
                    "choices": [],
                }
            )
        return samples

    return _make


@pytest.fixture
def media_assets(tmp_path_factory) -> Path:
    """Create temporary media files for multimodal tests."""

    root = tmp_path_factory.mktemp("media")
    (root / "dummy.jpg").write_bytes(base64.b64decode("/9j/4AAQSkZJRgABAQAAAQABAAD/"))
    (root / "dummy.wav").write_bytes(b"RIFF....WAVEfmt ")
    (root / "dummy.mp4").write_bytes(b"\x00\x00\x00 ftypmp42")
    (root / "dummy.pdf").write_bytes(b"%PDF-1.4")
    return root

