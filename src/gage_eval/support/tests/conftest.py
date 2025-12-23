from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest


@pytest.fixture
def temp_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a temp workspace and isolate GAGE_EVAL_SAVE_DIR."""

    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def dummy_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary dev_docs dataset directory."""

    dataset_dir = tmp_path / "dev_docs" / "dummy_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


@pytest.fixture
def dummy_jsonl(tmp_path: Path) -> Path:
    """Write a small dummy JSONL file for inspector tests."""

    path = tmp_path / "dummy.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id": "1", "question": "q1", "answer": "a1"}',
                '{"id": "2", "question": "q2", "answer": "a2"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def mock_agent() -> Callable[[str], str]:
    """Return a callable that simulates an Agent output."""

    def _call(prompt: str) -> str:
        return f"### FILE: output.txt\n{prompt}\n### END\n"

    return _call


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    """Minimal in-memory sample records."""

    return [
        {"id": "s1", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
        {"id": "s2", "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]},
    ]

