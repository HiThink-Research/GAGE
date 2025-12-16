from __future__ import annotations

import os
from pathlib import Path


def test_temp_workspace_sets_env(temp_workspace: Path) -> None:
    assert os.environ.get("GAGE_EVAL_SAVE_DIR") == str(temp_workspace)


def test_dummy_dataset_dir_exists(dummy_dataset_dir: Path) -> None:
    assert dummy_dataset_dir.exists()
    assert dummy_dataset_dir.name == "dummy_dataset"


def test_dummy_jsonl_written(dummy_jsonl: Path) -> None:
    text = dummy_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 2


def test_mock_agent_protocol(mock_agent) -> None:
    out = mock_agent("hello")
    assert out.startswith("### FILE:")
    assert out.strip().endswith("### END")

