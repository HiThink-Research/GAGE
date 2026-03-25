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


@pytest.mark.io
def test_jsonl_loader_default_preprocess_yields_transformed_samples(tmp_path: Path) -> None:
    dataset_path = tmp_path / "demo_echo.jsonl"
    dataset_path.write_text('{"id":"echo-1","prompt":"Hello, gage-eval!"}\n', encoding="utf-8")
    spec = DatasetSpec(
        dataset_id="demo_echo_dataset",
        loader="jsonl",
        params={"path": str(dataset_path)},
    )
    loader = JSONLDatasetLoader(spec)

    source = loader.load(None)
    records = list(source.records)

    assert len(records) == 1
    sample = records[0]

    assert sample.prompt == "Hello, gage-eval!"
    assert sample.inputs["prompt"] == "Hello, gage-eval!"
    assert sample.messages[0].role == "user"
    assert sample.messages[0].content[0].text == "Hello, gage-eval!"
