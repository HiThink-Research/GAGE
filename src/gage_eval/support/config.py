from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class AgentConfig:
    type: str = "gemini"  # gemini|codex
    command: str = "gemini"
    yolo_args: List[str] = field(default_factory=lambda: ["-y"])
    # Optional fine-grained Yolo configs for different agent CLIs.
    yolo_agent: Optional[str] = None  # e.g. "coding"
    model: Optional[str] = None  # e.g. "gpt-5.2" / "gemini-2.0-flash"
    reasoning_effort: Optional[str] = None  # low|medium|high (codex legacy client only)
    timeout: int = 900
    prompt_flag: str = "-m"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        # manual parsing to avoid dacite dependency
        return cls(
            type=data.get("type", "gemini"),
            command=data.get("command", "gemini"),
            yolo_args=data.get("yolo_args", ["-y"]),
            yolo_agent=data.get("yolo_agent"),
            model=data.get("model"),
            reasoning_effort=data.get("reasoning_effort"),
            timeout=data.get("timeout", 900),
            prompt_flag=data.get("prompt_flag", "-m"),
        )


    def build_yolo_args(self) -> List[str]:
        """Build final yolo args with structured overrides.

        Precedence:
          1) explicit raw flags in yolo_args
          2) structured fields (yolo_agent/model/reasoning_effort)
        """

        args = list(self.yolo_args)

        def _has(flag: str) -> bool:
            return any(a == flag or a.startswith(flag + "=") for a in args)

        if self.yolo_agent and not _has("--agent"):
            args += ["--agent", self.yolo_agent]
        if self.model and not _has("--model"):
            args += ["--model", self.model]
        if self.reasoning_effort and self.type == "codex":
            # Support both kebab/snake variants if users already set raw flags.
            if not _has("--reasoning-effort") and not _has("--reasoning_effort"):
                args += ["--reasoning-effort", self.reasoning_effort]
        return args


@dataclass
class PathConfig:
    workspace_root: Path = Path("dev_docs")
    local_datasets_root: Path = Path("local-datasets")
    # Where to write generated assets (src/config). Defaults to auto-detected gage-eval project root.
    project_root: Path = Path(".")


@dataclass
class ExecutionConfig:
    dry_run_default: bool = True
    command_allowlist: List[str] = field(default_factory=lambda: ["python", "pytest", "compileall"])


@dataclass
class SupportConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    language: str = "zh"  # zh|en

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SupportConfig":
        agent = AgentConfig.from_dict(data.get("agent") or {})
        paths_raw = data.get("paths") or {}
        paths = PathConfig(
            workspace_root=Path(paths_raw.get("workspace_root", PathConfig.workspace_root)),
            local_datasets_root=Path(paths_raw.get("local_datasets_root", PathConfig.local_datasets_root)),
            project_root=Path(paths_raw.get("project_root", PathConfig.project_root)),
        )
        execution = ExecutionConfig(**(data.get("execution") or {}))
        language = data.get("language", cls.language)
        if language not in ("zh", "en"):
            raise ValueError("language must be 'zh' or 'en'")
        return cls(agent=agent, paths=paths, execution=execution, language=language)


def _detect_project_root(cwd: Path) -> Path:
    """Detect gage-eval project root for writing src/config assets.

    Supports both:
      - running inside `gage-eval-main/`
      - running at a mono-repo root that contains `gage-eval-main/`
    """

    candidates = [cwd] + list(cwd.parents)
    for base in candidates:
        if (base / "src" / "gage_eval").exists() and (base / "config").exists():
            return base
        nested = base / "gage-eval-main"
        if (nested / "src" / "gage_eval").exists() and (nested / "config").exists():
            return nested
    return cwd


def _default_config_path() -> Path:
    return Path(".gage") / "support.yaml"


def _package_default_config_path() -> Path:
    """Default config shipped with the support module."""

    return Path(__file__).parent / "support.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(*, config_path: Optional[Path] = None) -> SupportConfig:
    """Load project-level Bench-Support config.

    Priority:
      1) explicit --config path
      2) .gage/support.yaml under cwd
      3) support/support.yaml shipped with module
      4) hard-coded defaults (defensive)
    """

    base_data: Dict[str, Any] = {}
    default_path = _package_default_config_path()
    if default_path.exists():
        loaded = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid default support config at {default_path}")
        base_data = loaded

    override_path = config_path or _default_config_path()
    if override_path and override_path.exists():
        loaded = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid support config at {override_path}")
        base_data = _deep_merge(base_data, loaded)

    cfg = SupportConfig.from_dict(base_data) if base_data else SupportConfig()

    # Auto-detect project root when not explicitly configured.
    if cfg.paths.project_root == Path("."):
        cfg.paths.project_root = _detect_project_root(Path.cwd())
    elif not cfg.paths.project_root.is_absolute():
        cfg.paths.project_root = (Path.cwd() / cfg.paths.project_root).resolve()

    return cfg


__all__ = ["SupportConfig", "AgentConfig", "PathConfig", "ExecutionConfig", "load_config"]
