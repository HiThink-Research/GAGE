from pathlib import Path
import importlib.util

import pytest


_RUN_PATH = Path(__file__).resolve().parents[2] / "run.py"
_SPEC = importlib.util.spec_from_file_location("gage_eval_run_cli", _RUN_PATH)
assert _SPEC and _SPEC.loader
run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run)


def test_expand_env_required_placeholder_uses_env_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    expanded = run._expand_env({"api_key": "${OPENAI_API_KEY:?set OPENAI_API_KEY}"})

    assert expanded == {"api_key": "sk-test"}


def test_expand_env_required_placeholder_reports_message(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="set OPENAI_API_KEY"):
        run._expand_env({"api_key": "${OPENAI_API_KEY:?set OPENAI_API_KEY}"})
