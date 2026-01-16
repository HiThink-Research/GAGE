from pathlib import Path

import pytest

from gage_eval.assets.datasets.loaders.jsonl_loader import JSONLDatasetLoader
from gage_eval.config.pipeline_config import DatasetSpec


@pytest.mark.io
def test_jsonl_loader_missing_appworld_hint(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_appworld.jsonl"
    spec = DatasetSpec(
        dataset_id="appworld_test",
        loader="jsonl",
        params={
            "path": str(missing_path),
            "missing_hint": "Run export_datasets.sh before loading AppWorld JSONL.",
        },
    )
    loader = JSONLDatasetLoader(spec)

    with pytest.raises(FileNotFoundError) as exc:
        loader.load(None)

    message = str(exc.value)
    assert "export_datasets.sh" in message
