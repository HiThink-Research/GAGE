from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "tau2" / "runtime_samples"
EXTRACTOR_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "tau2" / "_extract_0421_tau_fixtures.py"
CANONICAL_JSON_FIXTURES = {
    "0421_tau_old_failed_cost.json",
    "0421_tau2_unknown_user_side_tool_error.json",
}
RAW_TEXT_FIXTURES = {
    "0421_gemma4_airline_bare_call_respond_response.txt",
}


@pytest.mark.fast
def test_tau2_json_fixture_manifest_hashes_use_canonical_json() -> None:
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    entries = {
        entry["file"]: entry
        for entry in manifest.get("fixtures", [])
        if entry.get("file") in CANONICAL_JSON_FIXTURES
    }

    assert set(entries) == CANONICAL_JSON_FIXTURES
    for file_name, entry in entries.items():
        payload = json.loads((FIXTURE_DIR / file_name).read_text(encoding="utf-8"))
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        assert entry["sha256"] == digest


@pytest.mark.fast
def test_tau2_text_fixture_manifest_hashes_use_raw_bytes() -> None:
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    entries = {
        entry["file"]: entry
        for entry in manifest.get("fixtures", [])
        if entry.get("file") in RAW_TEXT_FIXTURES
    }

    assert set(entries) == RAW_TEXT_FIXTURES
    for file_name, entry in entries.items():
        payload = (FIXTURE_DIR / file_name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        assert entry["sha256"] == digest


@pytest.mark.fast
def test_tau2_fixture_extractor_declares_new_tau2_fixtures() -> None:
    spec = importlib.util.spec_from_file_location("extract_0421_tau_fixtures", EXTRACTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    declared_outputs = {
        item["output"]
        for collection_name in ("TRACE_FIXTURES", "JSON_FIXTURES", "ARTIFACT_FIXTURES")
        for item in getattr(module, collection_name, [])
    }

    assert "0421_tau2_think_tail_bare_json_respond_response.txt" in declared_outputs
    assert "0421_gemma4_airline_bare_call_respond_response.txt" in declared_outputs
    assert "0421_tau2_unknown_user_side_tool_error.json" in declared_outputs
