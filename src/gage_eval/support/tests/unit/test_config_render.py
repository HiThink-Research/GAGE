from __future__ import annotations

from pathlib import Path

import yaml

from gage_eval.support.pipeline import _render_pipeline_configs


def test_render_pipeline_configs_minimal() -> None:
    cfg_block = {
        "dataset_id": "dummy_ds",
        "preprocess_name": "dummy_preprocess",
        "fields": {
            "question_field": "question",
            "answers_field": "answer",
            "content_field": "messages",
        },
        "metrics": [{"metric_id": "exact_match", "implementation": "exact_match"}],
    }
    rendered = _render_pipeline_configs(cfg_block, slug="dummy_ds")
    for kind, text in rendered.items():
        data = yaml.safe_load(text)
        assert data["api_version"] == "gage/v1alpha1"
        assert data["kind"] == "PipelineConfig"
        assert data["datasets"][0]["params"]["preprocess"] == "dummy_preprocess"

