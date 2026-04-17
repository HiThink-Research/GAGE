from __future__ import annotations

from pathlib import Path

import datasets

from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.assets.datasets.loaders.hf_hub_loader import (
    _build_local_data_files_arg,
    _load_local_dataset,
    _maybe_prioritize_local_smoke_subset,
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


def test_prioritize_local_smoke_subset_reorders_before_limit(tmp_path: Path) -> None:
    records = datasets.Dataset.from_list(
        [
            {"instance_id": "non_smoke__repo-1", "problem_statement": "skip"},
            {"instance_id": "smoke__repo-1", "problem_statement": "keep"},
            {"instance_id": "smoke__repo-2", "problem_statement": "keep-too"},
        ]
    )
    smoke_ids_path = tmp_path / "smoke_instance_ids.txt"
    smoke_ids_path.write_text("smoke__repo-2\nsmoke__repo-1\n", encoding="utf-8")
    spec = DatasetSpec(
        dataset_id="swebench_local_smoke",
        hub="huggingface",
        loader="hf_hub",
        params={"preprocess_kwargs": {"smoke_ids_path": str(smoke_ids_path)}},
    )

    reordered = _maybe_prioritize_local_smoke_subset(
        records,
        spec,
        local_path=str(tmp_path),
    )
    limited = reordered.select([0])

    assert limited[0]["instance_id"] == "smoke__repo-2"
