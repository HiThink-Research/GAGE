from __future__ import annotations

import hashlib
import json

import pytest

from gage_eval.external_harness_kits.secret_redaction import (
    SecretRedactionContext,
    redact_for_artifact,
    redact_text,
    to_invocation_artifact,
)


@pytest.mark.fast
def test_resolved_vllm_api_key_enters_subprocess_env_but_not_invocation_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "plain-vllm-secret-value"
    monkeypatch.setenv("VLLM_API_KEY", secret)
    subprocess_env = {"VLLM_API_KEY": secret, "TRACE_LEVEL": "debug"}
    context = SecretRedactionContext.from_environ(subprocess_env)
    invocation = {
        "job_name": "gage_terminal_bench_lmstudio",
        "launcher_mode": "python_subprocess",
        "launcher_argv": ["python", "-m", "gage_eval.external_harness_kits.harbor.launcher"],
        "environ": subprocess_env,
        "job_config": {
            "agents": [
                {
                    "name": "terminus-2",
                    "kwargs": {
                        "api_base": "http://127.0.0.1:1234/v1",
                        "model_id": "qwen/qwen3.5-9b",
                    },
                }
            ]
        },
    }

    artifact = to_invocation_artifact(invocation, context=context)

    assert subprocess_env["VLLM_API_KEY"] == secret
    artifact_json = json.dumps(artifact, sort_keys=True)
    assert secret not in artifact_json
    assert artifact["environ"] == {
        "TRACE_LEVEL": {
            "is_secret": False,
            "value": "<omitted>",
            "value_sha256": _sha256("debug"),
        },
        "VLLM_API_KEY": {
            "is_secret": True,
            "value": "<redacted>",
            "value_sha256": _sha256(secret),
        },
    }
    assert "debug" not in artifact_json


@pytest.mark.fast
def test_invocation_artifact_preserves_non_secret_model_fields() -> None:
    secret = "plain-vllm-secret-value"
    context = SecretRedactionContext(secret_values={secret})
    invocation = {
        "job_name": "gage_terminal_bench_lmstudio",
        "environ": {"VLLM_API_KEY": secret},
        "job_config": {
            "agents": [
                {
                    "name": "terminus-2",
                    "kwargs": {
                        "api_base": "http://127.0.0.1:1234/v1",
                        "model_id": "qwen/qwen3.5-9b",
                    },
                }
            ]
        },
    }

    artifact = to_invocation_artifact(invocation, context=context)

    kwargs = artifact["job_config"]["agents"][0]["kwargs"]
    assert kwargs["api_base"] == "http://127.0.0.1:1234/v1"
    assert kwargs["model_id"] == "qwen/qwen3.5-9b"


@pytest.mark.fast
def test_nested_agent_kwargs_redact_by_secret_key_and_resolved_value() -> None:
    secret = "plain-vllm-secret-value"
    context = SecretRedactionContext(secret_values={secret})
    payload = {
        "kwargs": {
            "api_key": "not-the-resolved-secret",
            "headers": {
                "authorization": "Bearer plain-header-secret",
                "x-request-id": secret,
            },
            "nested": [{"session_token": "plain-session-token"}],
            "api_base": "http://127.0.0.1:1234/v1",
        }
    }

    redacted = redact_for_artifact(payload, context=context)

    assert redacted["kwargs"]["api_key"] == "<redacted>"
    assert redacted["kwargs"]["headers"]["authorization"] == "<redacted>"
    assert redacted["kwargs"]["headers"]["x-request-id"] == "<redacted>"
    assert redacted["kwargs"]["nested"][0]["session_token"] == "<redacted>"
    assert redacted["kwargs"]["api_base"] == "http://127.0.0.1:1234/v1"
    redacted_json = json.dumps(redacted)
    assert secret not in redacted_json
    assert "plain-header-secret" not in redacted_json
    assert "plain-session-token" not in redacted_json


@pytest.mark.fast
def test_redact_text_uses_resolved_value_as_second_line_of_defense() -> None:
    context = SecretRedactionContext(secret_values={"plain-vllm-secret-value"})

    assert redact_text("error leaked plain-vllm-secret-value", context=context) == "error leaked <redacted>"
    assert redact_text("Authorization: Bearer abc123", context=context) == "Authorization: Bearer <redacted>"
    assert redact_text("literal ${VLLM_API_KEY}", context=context) == "literal <redacted>"


@pytest.mark.fast
def test_redact_text_reads_secret_values_from_current_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "plain-vllm-secret-value"
    monkeypatch.setenv("VLLM_API_KEY", secret)

    assert redact_text(f"launcher traceback leaked {secret}") == "launcher traceback leaked <redacted>"


@pytest.mark.fast
def test_redact_text_removes_common_credential_literals_without_env_context() -> None:
    assert redact_text("api_key=plain-vllm-secret-value", environ={}) == "api_key=<redacted>"
    assert redact_text("token: plain-session-token-value", environ={}) == "token: <redacted>"
    assert redact_text('password="plain-password-value"', environ={}) == 'password="<redacted>"'
    assert redact_text("authorization: Basic dXNlcjpwYXNz", environ={}) == "authorization: Basic <redacted>"
    assert redact_text("Authorization=ApiKey abc.def", environ={}) == "Authorization=ApiKey <redacted>"


@pytest.mark.fast
def test_redact_for_artifact_handles_camel_case_secret_keys() -> None:
    redacted = redact_for_artifact(
        {
            "accessToken": "plain-access-token-value",
            "clientSecret": "plain-client-secret-value",
            "modelInfo": {"maxInputTokens": 8192},
        }
    )

    assert redacted["accessToken"] == "<redacted>"
    assert redacted["clientSecret"] == "<redacted>"
    assert redacted["modelInfo"]["maxInputTokens"] == 8192


def _sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
