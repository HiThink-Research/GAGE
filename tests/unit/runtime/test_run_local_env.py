from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import run as gage_run


@pytest.mark.fast
def test_detect_workspace_root_prefers_parent_workspace(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (tmp_path / "env").mkdir()
    (tmp_path / "env" / "localenv").write_text('OPENAI_API_KEY="from-parent"\n', encoding="utf-8")

    assert gage_run._detect_workspace_root(repo_root) == tmp_path


@pytest.mark.fast
def test_load_local_env_file_parses_export_and_preserves_existing_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / "localenv"
    env_file.write_text(
        'OPENAI_API_KEY="from-file"\nexport GH_TOKEN="gh-token"\nEMPTY=\n# comment\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "existing-key")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)

    loaded = gage_run._load_local_env_file(env_file)

    assert os.environ["OPENAI_API_KEY"] == "existing-key"
    assert os.environ["GH_TOKEN"] == "gh-token"
    assert os.environ["EMPTY"] == ""
    assert loaded == {"GH_TOKEN": "gh-token", "EMPTY": ""}


@pytest.mark.fast
def test_load_workspace_local_env_layers_run_env_then_localenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    scripts_dir = tmp_path / "env" / "scripts"
    scripts_dir.mkdir(parents=True)
    run_env = scripts_dir / "run.env"
    local_env = tmp_path / "env" / "localenv"
    run_env.write_text('OPENAI_API_KEY="from-run"\nBASE_URL="from-run"\n', encoding="utf-8")
    local_env.write_text('OPENAI_API_KEY="from-local"\nali_api_key="ali-token"\n', encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.delenv("ali_api_key", raising=False)
    monkeypatch.delenv("GAGE_LOCAL_ENV_FILE", raising=False)
    monkeypatch.delenv("GAGE_WORKSPACE_ROOT", raising=False)

    loaded_files = gage_run._load_workspace_local_env(repo_root)

    assert loaded_files == [run_env, local_env]
    assert os.environ["OPENAI_API_KEY"] == "from-local"
    assert os.environ["BASE_URL"] == "from-run"
    assert os.environ["ali_api_key"] == "ali-token"
    assert os.environ["GAGE_WORKSPACE_ROOT"] == str(tmp_path)


@pytest.mark.fast
def test_load_workspace_local_env_preserves_explicit_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    scripts_dir = tmp_path / "env" / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "run.env").write_text('OPENAI_API_KEY="from-run"\n', encoding="utf-8")
    (tmp_path / "env" / "localenv").write_text(
        'OPENAI_API_KEY="from-local"\nali_api_key="ali-token"\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    monkeypatch.delenv("ali_api_key", raising=False)
    monkeypatch.delenv("GAGE_LOCAL_ENV_FILE", raising=False)
    monkeypatch.delenv("GAGE_WORKSPACE_ROOT", raising=False)

    gage_run._load_workspace_local_env(repo_root)

    assert os.environ["OPENAI_API_KEY"] == "from-shell"
    assert os.environ["ali_api_key"] == "ali-token"
