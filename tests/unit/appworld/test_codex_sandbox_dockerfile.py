from __future__ import annotations

from pathlib import Path


def test_codex_sandbox_dockerfile_installs_codex() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[3]
        / "docker"
        / "agent_eval"
        / "codex_sandbox"
        / "Dockerfile"
    )
    content = dockerfile.read_text(encoding="utf-8")

    assert "npm install -g @openai/codex@latest" in content
    assert "CODEX_HOME=/agent" in content
    assert "WORKDIR /workspace" in content
