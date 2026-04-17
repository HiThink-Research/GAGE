from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.io
@pytest.mark.parametrize(
    ("config_path", "expected_backend_id"),
    (
        ("config/custom/charxiv/charxiv_vllm_async_chat.yaml", "litellm_backend"),
        ("config/custom/mmlu_pro/mmlu_pro_chat.yaml", "litellm_backend"),
    ),
)
def test_chat_eval_configs_keep_backends_outside_dataset_list(
    config_path: str,
    expected_backend_id: str,
) -> None:
    root = Path(__file__).resolve().parents[3]
    path = root / config_path
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    config = PipelineConfig.from_dict(payload)

    assert all(spec.dataset_id for spec in config.datasets)
    assert any(spec.backend_id == expected_backend_id for spec in config.backends)
