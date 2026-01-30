from __future__ import annotations

import pytest

from gage_eval.assets.datasets.loaders.loader_utils import apply_bundle
from gage_eval.config.pipeline_config import DatasetSpec


@pytest.mark.fast
def test_apply_bundle_pass_through_without_bundle() -> None:
    spec = DatasetSpec(dataset_id="ds_no_bundle", loader="hf_hub", params={})
    records = [{"id": "x1", "field": "value"}]

    output = list(apply_bundle(records, spec, data_path="dummy"))

    assert output == records
