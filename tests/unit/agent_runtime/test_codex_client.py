from __future__ import annotations

from pathlib import Path

import pytest

from gage_eval.agent_runtime.clients.codex import CodexClient
from gage_eval.agent_runtime.clients.types import ClientRunRequest


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = str(payload)

    def json(self) -> dict:
        return dict(self._payload)


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def post(self, url: str, json: dict, headers: dict[str, str], timeout: int) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return self.response


class _TraceStub:
    def to_dict(self) -> dict[str, object]:
        return {"trace_id": "trace-1"}


def test_codex_client_calls_service_and_persists_runtime_artifacts(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample"
    session = _FakeSession(
        _FakeResponse(
            status_code=200,
            payload={
                "result": {
                    "exit_code": 0,
                    "stdout": "final message",
                    "stderr": "",
                    "answer": "final message",
                    "patch_content": "diff --git a/a b/a\n",
                    "usage": {"total_tokens": 1234},
                    "agent_trace": [{"tool": "spotify_search", "status": "ok"}],
                    "metadata": {"provider_name": "codex-service"},
                }
            },
            headers={"x-request-id": "req-123"},
        )
    )
    client = CodexClient(service_url="http://codex-service.local", session=session)

    result = client.run(
        ClientRunRequest(
            instruction="Fix the task and stop.",
            cwd="/workspace/repo",
            metadata={"submission_contract": "submission.patch"},
        ),
        {
            "artifact_layout": {
                "sample_root": str(sample_root),
                "artifacts_dir": str(sample_root / "artifacts"),
            },
            "runtime_context": {"mcp_endpoint": "http://127.0.0.1:5001"},
            "sandbox_provider": object(),
        },
    )

    assert session.calls
    assert session.calls[0]["url"] == "http://codex-service.local/run"
    assert "sandbox_provider" not in session.calls[0]["json"]["environment"]
    assert session.calls[0]["headers"]["Content-Type"] == "application/json"
    assert result.stdout == "final message"
    assert result.patch_path == str(sample_root / "submission.patch")
    assert result.patch_content == "diff --git a/a b/a\n"
    assert result.usage == {"total_tokens": 1234}
    assert result.trajectory_path == str(sample_root / "artifacts" / "trajectory.log")
    assert result.artifacts["stdout"] == str(sample_root / "artifacts" / "stdout.log")
    assert result.metadata["service_url"] == "http://codex-service.local/run"
    assert result.metadata["request_id"] == "req-123"
    assert Path(result.patch_path).read_text(encoding="utf-8").startswith("diff --git")
    assert "spotify_search" in Path(result.trajectory_path).read_text(encoding="utf-8")


def test_codex_client_reads_service_url_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GAGE_CODEX_CLIENT_URL", "http://127.0.0.1:8765")
    sample_root = tmp_path / "sample"
    session = _FakeSession(_FakeResponse(status_code=200, payload={"result": {"exit_code": 0}}))
    client = CodexClient(session=session)

    client.run(
        ClientRunRequest(instruction="reply ok"),
        {
            "artifact_layout": {
                "sample_root": str(sample_root),
                "artifacts_dir": str(sample_root / "artifacts"),
            }
        },
    )

    assert session.calls[0]["url"] == "http://127.0.0.1:8765/run"


def test_codex_client_adds_bearer_token_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GAGE_CODEX_CLIENT_TOKEN", "secret-token")
    sample_root = tmp_path / "sample"
    session = _FakeSession(_FakeResponse(status_code=200, payload={"result": {"exit_code": 0}}))
    client = CodexClient(service_url="http://127.0.0.1:8765", session=session)

    client.run(
        ClientRunRequest(instruction="reply ok"),
        {
            "artifact_layout": {
                "sample_root": str(sample_root),
                "artifacts_dir": str(sample_root / "artifacts"),
            }
        },
    )

    assert session.calls[0]["headers"]["Authorization"] == "Bearer secret-token"


def test_codex_client_serializes_non_json_request_payloads(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample"
    session = _FakeSession(_FakeResponse(status_code=200, payload={"result": {"exit_code": 0}}))
    client = CodexClient(service_url="http://codex-service.local", session=session)

    client.run(
        ClientRunRequest(
            instruction="reply ok",
            metadata={"trace": _TraceStub()},
            payload={"trace": _TraceStub()},
        ),
        {
            "artifact_layout": {
                "sample_root": str(sample_root),
                "artifacts_dir": str(sample_root / "artifacts"),
            }
        },
    )

    request_payload = session.calls[0]["json"]["request"]
    assert request_payload["metadata"]["trace"]["trace_id"] == "trace-1"
    assert request_payload["payload"]["trace"]["trace_id"] == "trace-1"


def test_codex_client_raises_on_service_http_error(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample"
    session = _FakeSession(_FakeResponse(status_code=503, payload={"error": "unavailable"}))
    client = CodexClient(service_url="http://codex-service.local/run", session=session)

    with pytest.raises(RuntimeError, match="codex_service_http_error:503"):
        client.run(
            ClientRunRequest(instruction="reply ok"),
            {
                "artifact_layout": {
                    "sample_root": str(sample_root),
                    "artifacts_dir": str(sample_root / "artifacts"),
                }
            },
        )
