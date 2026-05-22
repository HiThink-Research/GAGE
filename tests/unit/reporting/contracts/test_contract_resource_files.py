from __future__ import annotations

from importlib import resources

import pytest
import yaml


@pytest.mark.fast
def test_machine_schema_yaml_files_are_package_resources() -> None:
    contract_files = resources.files("gage_eval.reporting.contracts")

    expected = {
        "report_context_headline.yaml": {
            "schema_version": "gage.report_context_headline.v1",
            "required_key": "required_fields",
            "required_value": "verdict",
        },
        "summary_v1_compat_fields.yaml": {
            "schema_version": "gage.summary_v1_compat_fields.v1",
            "required_key": "required_fields",
            "required_value": "runtime_health",
        },
    }

    for filename, expectation in expected.items():
        resource = contract_files / filename
        assert resource.is_file()
        payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
        assert payload["schema_version"] == expectation["schema_version"]
        assert expectation["required_value"] in payload[expectation["required_key"]]
