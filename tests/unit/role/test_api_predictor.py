from __future__ import annotations

import pytest

from gage_eval.evaluation.support_artifacts import record_support_output
from gage_eval.role.adapters.helper_model import HelperModelAdapter


@pytest.mark.fast
def test_api_predictor_emits_filters() -> None:
    adapter = HelperModelAdapter(
        adapter_id="api_predictor",
        backend=lambda _: {},
        implementation="appworld_api_predictor",
        implementation_params={
            "always_include_apps": ["api_docs", "supervisor"],
            "always_include_tools": ["supervisor.complete_task"],
        },
    )
    response = {"answer": "spotify.login\nvenmo__pay\nspotify.login"}
    payload = {
        "sample": {
            "support_outputs": [
                {
                    "api_descriptions_allowed_apis": ["spotify.login"],
                    "api_descriptions_allowed_tools": ["spotify__login", "supervisor__complete_task"],
                }
            ]
        }
    }

    result = adapter.handle_backend_response(payload, response)

    assert result["predicted_apis"] == ["spotify.login"]
    assert result["tool_allowlist"] == [
        "spotify__login",
        "supervisor__complete_task",
    ]
    assert result["tool_prefixes"] == ["supervisor"]
    assert result["tool_doc_allowed_apps"] == ["spotify", "supervisor"]
    event = result["observability_events"][0]
    assert event["event"] == "api_predictor_result"
    assert event["payload"]["predicted_count"] == 1


@pytest.mark.fast
def test_api_predictor_reads_allowed_info_from_support_artifacts() -> None:
    adapter = HelperModelAdapter(
        adapter_id="api_predictor",
        backend=lambda _: {},
        implementation="appworld_api_predictor",
    )
    sample = {}
    record_support_output(
        sample,
        slot_id="support:00:api_docs",
        adapter_id="api_docs",
        output={
            "api_descriptions_context": "Spotify APIs",
            "api_descriptions_allowed_apis": ["spotify.login"],
            "api_descriptions_allowed_tools": ["spotify__login"],
        },
    )
    sample.pop("support_outputs", None)

    result = adapter.handle_backend_response(
        {"sample": sample},
        {"answer": "spotify.login\nvenmo.pay"},
    )

    assert result["predicted_apis"] == ["spotify.login"]
    assert result["tool_allowlist"] == ["spotify__login"]
