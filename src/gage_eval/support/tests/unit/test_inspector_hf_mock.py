from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from gage_eval.support.config import PathConfig, SupportConfig
from gage_eval.support.inspector import inspect_dataset


class _FakeDataset:
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._records[idx]


def test_inspect_hf_dataset_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]

    def fake_load_dataset(*args, **kwargs):
        return _FakeDataset(records)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    cfg = SupportConfig(paths=PathConfig(workspace_root=tmp_path / "dev_docs", local_datasets_root=tmp_path))
    out_dir = inspect_dataset(
        dataset_name="dummy/hf",
        subset=None,
        split="train",
        max_samples=2,
        local_path=None,
        cfg=cfg,
    )
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["hub_id"] == "dummy/hf"
    sample = json.loads((out_dir / "sample.json").read_text(encoding="utf-8"))
    assert len(sample) == 2
