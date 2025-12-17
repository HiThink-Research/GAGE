from __future__ import annotations

from pathlib import Path

import pytest

from gage_eval.support.utils import detect_state


def test_detect_state_progression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_dir = tmp_path / "dev_docs" / "foo"
    dataset_dir.mkdir(parents=True)

    # Pending -> Inspected
    (dataset_dir / "sample.json").write_text("[]", encoding="utf-8")
    assert detect_state(dataset_dir) == "Inspected"

    # Inspected -> Designed
    (dataset_dir / "design.md").write_text(
        "```yaml support_config\ndataset_id: foo\npreprocess_name: foo_p\nfields: {}\n```\n",
        encoding="utf-8",
    )
    assert detect_state(dataset_dir) == "Designed"

    # Designed -> Implemented when assets exist
    preproc_dir = tmp_path / "src" / "gage_eval" / "assets" / "datasets" / "preprocessors"
    preproc_dir.mkdir(parents=True)
    (preproc_dir / "foo_preprocessor.py").write_text("class FooPreprocessor: ...\n", encoding="utf-8")

    config_dir = tmp_path / "config" / "custom"
    config_dir.mkdir(parents=True)
    (config_dir / "foo_openai.yaml").write_text("api_version: gage/v1alpha1\nkind: PipelineConfig\n", encoding="utf-8")

    assert detect_state(dataset_dir) == "Implemented"

