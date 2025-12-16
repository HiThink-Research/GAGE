from __future__ import annotations

import json
from pathlib import Path

from gage_eval.support.config import PathConfig, SupportConfig
from gage_eval.support.inspector import inspect_dataset


def test_inspect_local_jsonl(tmp_path: Path, dummy_jsonl: Path) -> None:
    cfg = SupportConfig(paths=PathConfig(workspace_root=tmp_path / "dev_docs", local_datasets_root=tmp_path))
    out_dir = inspect_dataset(
        dataset_name=str(dummy_jsonl),
        subset=None,
        split=None,
        max_samples=1,
        local_path=dummy_jsonl,
        cfg=cfg,
    )
    assert (out_dir / "meta.json").exists()
    assert (out_dir / "sample.json").exists()
    assert (out_dir / "schema.json").exists()
    sample = json.loads((out_dir / "sample.json").read_text(encoding="utf-8"))
    assert len(sample) == 1

