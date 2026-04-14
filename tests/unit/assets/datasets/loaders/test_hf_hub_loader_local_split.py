from __future__ import annotations

from pathlib import Path

import datasets

from gage_eval.assets.datasets.loaders.hf_hub_loader import (
    _build_local_data_files_arg,
    _load_local_dataset,
)


def test_build_local_data_files_arg_preserves_named_split_files() -> None:
    files = [
        "/tmp/swebench/data/test-00000-of-00001.parquet",
        "/tmp/swebench/data/test-00001-of-00001.parquet",
    ]

    data_files = _build_local_data_files_arg(files, split="test")

    assert data_files == {"test": files}


def test_load_local_dataset_supports_named_test_split(tmp_path: Path) -> None:
    dataset_root = tmp_path / "swebench_local"
    data_dir = dataset_root / "data"
    data_dir.mkdir(parents=True)

    records_path = data_dir / "test-00000-of-00001.jsonl"
    records_path.write_text('{"instance_id":"demo__repo-1","problem_statement":"Fix it"}\n', encoding="utf-8")

    dataset = _load_local_dataset(
        datasets_module=datasets,
        local_path=str(dataset_root),
        split="test",
        subset=None,
        load_kwargs={},
        builder_name=None,
    )

    row = dataset[0]
    assert row["instance_id"] == "demo__repo-1"
    assert row["problem_statement"] == "Fix it"
