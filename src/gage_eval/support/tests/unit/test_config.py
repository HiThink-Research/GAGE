from __future__ import annotations

from pathlib import Path

import pytest

from gage_eval.support.config import load_config


def test_load_defaults_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.language == "zh"
    assert cfg.agent.type in ("gemini", "codex")


def test_load_project_level_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / ".gage"
    project.mkdir()
    (project / "support.yaml").write_text(
        "language: en\nagent:\n  type: codex\npaths:\n  workspace_root: docs\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.language == "en"
    assert cfg.agent.type == "codex"
    assert str(cfg.paths.workspace_root) == "docs"


def test_explicit_config_path_overrides_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / ".gage"
    project.mkdir()
    (project / "support.yaml").write_text("language: zh\n", encoding="utf-8")
    explicit = tmp_path / "custom.yaml"
    explicit.write_text("language: en\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = load_config(config_path=explicit)
    assert cfg.language == "en"

