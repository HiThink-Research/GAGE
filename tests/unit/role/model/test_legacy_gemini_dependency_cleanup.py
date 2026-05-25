from __future__ import annotations

import json
from pathlib import Path

import pytest

from gage_eval.registry import registry
from gage_eval.role.model.backends import builder
from gage_eval.role.model.backends import vendor_http_backend


REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.io
def test_legacy_google_generativeai_backend_is_not_exposed() -> None:
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    manifest = json.loads(
        (REPO_ROOT / "src/gage_eval/registry/manifests/backends.json").read_text(encoding="utf-8")
    )

    assert "google-generativeai" not in requirements
    assert "gemini_http" not in builder._BACKEND_CONFIG_SCHEMAS
    assert not hasattr(vendor_http_backend, "GeminiHTTPBackend")
    assert all(entry["name"] != "gemini_http" for entry in manifest["entries"])
    assert all(entry.name != "gemini_http" for entry in registry.list("backends"))
