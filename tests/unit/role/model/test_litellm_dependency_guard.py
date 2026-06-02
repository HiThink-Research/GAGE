from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from gage_eval.role.model.config import litellm as litellm_config


@pytest.mark.fast
def test_litellm_capability_guard_reports_missing_router(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.SimpleNamespace(completion=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "litellm", fake)

    with pytest.raises(RuntimeError, match="LiteLLM Router"):
        litellm_config.assert_litellm_capabilities(require_router=True)


@pytest.mark.fast
def test_litellm_capability_guard_reports_missing_acompletion(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.SimpleNamespace(completion=lambda **kwargs: None, Router=object)
    monkeypatch.setitem(sys.modules, "litellm", fake)

    with pytest.raises(RuntimeError, match="LiteLLM acompletion"):
        litellm_config.assert_litellm_capabilities(require_async=True)


@pytest.mark.fast
def test_litellm_capability_guard_reports_missing_function_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.SimpleNamespace(completion=lambda **kwargs: None, Router=object, acompletion=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "litellm", fake)

    with pytest.raises(RuntimeError, match="LiteLLM supports_function_calling"):
        litellm_config.assert_litellm_capabilities(require_function_calling=True)


@pytest.mark.io
def test_litellm_and_vllm_requirements_match_design_baseline() -> None:
    requirements = (Path(__file__).resolve().parents[4] / "requirements.txt").read_text(encoding="utf-8")

    assert litellm_config.MIN_LITELLM_VERSION == "1.83.14"
    assert litellm_config.MIN_VLLM_VERSION == "0.20.2"
    assert f"litellm=={litellm_config.MIN_LITELLM_VERSION}" in requirements
    assert "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.20.2/" in requirements
    assert "openai>=2.20.0" in requirements
    assert "pydantic>=2.12.0" in requirements
    assert "litellm>=1.36.0" not in requirements
    assert "vllm>=0.4.0" not in requirements
