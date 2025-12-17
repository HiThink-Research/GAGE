from __future__ import annotations

import re
import subprocess
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from loguru import logger

from .config import SupportConfig


_SUPPORT_CONFIG_RE = re.compile(
    r"```yaml\s+support_config\s*\n(.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)

_ARTIFACT_EXT_RE = re.compile(r"\.(jsonl|json|csv|tsv|parquet|txt)$", flags=re.IGNORECASE)
_PREPROCESS_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def slugify_dataset_name(name: str) -> str:
    """Convert dataset name/path into a filesystem + import friendly slug.

    - Replace any non [0-9A-Za-z_] characters with underscore.
    - Collapse repeated underscores.
    """

    raw = str(name or "")
    raw = raw.replace("/", "_").replace("\\", "_")
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", raw)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if slug and slug[0].isdigit():
        slug = f"d_{slug}"
    return slug or "dataset"

def artifact_slug_from_dataset_id(dataset_id: str) -> str:
    """Derive a short, lower-case slug for artifact filenames/modules.

    Examples:
    - HuggingFaceH4/MATH-500 -> math500
    - /path/to/math500.jsonl -> math500
    """

    raw = str(dataset_id or "").strip()
    if not raw:
        return "dataset"

    # Prefer the last segment of hub-id-like strings or filesystem paths.
    tail = re.split(r"[\\/]", raw)[-1]
    tail = _ARTIFACT_EXT_RE.sub("", tail)

    # Remove separators and keep only [a-z0-9] to make it import-safe.
    slug = re.sub(r"[^0-9A-Za-z]+", "", tail).lower()
    if not slug:
        slug = slugify_dataset_name(raw).lower()
    slug = slug or "dataset"
    if slug[0].isdigit():
        slug = f"d{slug}"
    return slug


def normalize_preprocess_name(preprocess_name: str, *, artifact_slug: str) -> str:
    """Normalize support_config.preprocess_name to a short, stable registry id."""

    raw = str(preprocess_name or "").strip()
    if raw and raw == raw.lower() and len(raw) <= 32 and _PREPROCESS_NAME_RE.fullmatch(raw):
        return raw
    return artifact_slug


def parse_support_config(design_md: str) -> Dict[str, Any]:
    """Extract and parse the unique yaml support_config block from design.md."""

    matches = _SUPPORT_CONFIG_RE.findall(design_md)
    if len(matches) != 1:
        raise ValueError(f"design.md must contain exactly one yaml support_config block (found {len(matches)})")
    raw = matches[0]
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError("support_config block must be a YAML mapping")
    for key in ("dataset_id", "preprocess_name", "fields"):
        if key not in data:
            raise ValueError(f"support_config missing required field '{key}'")
    if not isinstance(data.get("fields"), dict):
        raise ValueError("support_config.fields must be a mapping")
    return data


def ensure_git_clean(force: bool = False, *, cwd: Optional[Path] = None) -> None:
    """Guard against overwriting with dirty git working tree."""

    if force:
        return
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            logger.warning("Not a git repo or git unavailable; skip git guard.")
            return
        if completed.stdout.strip():
            raise RuntimeError("Git working tree is dirty. Please commit/stash or rerun with --force.")
    except FileNotFoundError:
        logger.warning("git not found; skip git guard.")


_INJECTION_TOKENS = ("`", "$(", ";", "&&", "|", ">", "<")


def is_safe_command(command: str) -> bool:
    return not any(tok in command for tok in _INJECTION_TOKENS)


def iter_test_commands(cfg_block: Dict[str, Any]) -> List[str]:
    tests = cfg_block.get("tests") or {}
    cmds = tests.get("run_commands") or []
    return [str(c).strip() for c in cmds if str(c).strip()]


def guard_commands(commands: Iterable[str], cfg: SupportConfig, *, confirm: bool = True) -> List[str]:
    """Apply allowlist and injection guards. Return commands to execute."""

    env_assign_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")

    def _extract_prefix(cmd: str) -> str:
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()
        for tok in tokens:
            if env_assign_re.match(tok):
                continue
            return tok
        return tokens[0] if tokens else ""

    safe: List[str] = []
    allowlist = set(cfg.execution.command_allowlist)
    for cmd in commands:
        if not is_safe_command(cmd):
            raise RuntimeError(f"Unsafe command rejected: {cmd}")
        prefix = _extract_prefix(cmd)
        if prefix not in allowlist:
            if confirm:
                import typer

                if not typer.confirm(f"Command '{cmd}' not in allowlist. Execute?"):
                    continue
            else:
                continue
        safe.append(cmd)
    return safe


def detect_state(dataset_dir: Path, cfg: Optional[SupportConfig] = None) -> str:
    """Detect current state based on file existence."""

    if not dataset_dir.exists():
        return "Pending"
    if (dataset_dir / "sample.json").exists():
        state = "Inspected"
    else:
        return "Pending"
    if (dataset_dir / "design.md").exists():
        state = "Designed"

    slug_candidates: list[str] = []
    design_path = dataset_dir / "design.md"
    if design_path.exists():
        try:
            cfg_block = parse_support_config(design_path.read_text(encoding="utf-8"))
            artifact_slug = artifact_slug_from_dataset_id(str(cfg_block.get("dataset_id") or ""))
            normalized = normalize_preprocess_name(str(cfg_block.get("preprocess_name") or ""), artifact_slug=artifact_slug)
            if normalized:
                slug_candidates.append(normalized)
        except Exception:
            pass

    slug_candidates.append(slugify_dataset_name(dataset_dir.name))
    slug_candidates = [s for i, s in enumerate(slug_candidates) if s and s not in slug_candidates[:i]]

    project_root = (cfg.paths.project_root if cfg else Path("."))
    if not project_root.is_absolute():
        project_root = (Path.cwd() / project_root).resolve()

    preproc_root = project_root / "src/gage_eval/assets/datasets/preprocessors"
    config_root = project_root / "config/custom"
    preproc_exists = any((preproc_root / f"{s}_preprocessor.py").exists() for s in slug_candidates)
    config_exists = any(
        (config_root / f"{s}_openai.yaml").exists() or (config_root / f"{s}_vllm.yaml").exists()
        for s in slug_candidates
    )
    if preproc_exists and config_exists:
        state = "Implemented"
    return state


def show_status(dataset: Optional[str], cfg: SupportConfig, next_step: bool, auto: bool) -> None:
    """Print status for datasets."""

    import typer

    root = cfg.paths.workspace_root
    if dataset:
        dirs = [root / dataset]
    else:
        dirs = sorted([p for p in root.glob("*") if p.is_dir()])
    for d in dirs:
        st = detect_state(d, cfg=cfg)
        typer.echo(f"{d.name}: {st}")
        if next_step:
            ns = {"Pending": "inspect", "Inspected": "design", "Designed": "implement"}.get(st)
            if ns:
                typer.echo(f"  next: {ns}")
        if auto:
            ns = {"Pending": "inspect", "Inspected": "design", "Designed": "implement"}.get(st)
            if ns and typer.confirm(f"Perform next step '{ns}' for {d.name}?"):
                typer.echo("Auto mode not yet implemented.")


def class_name_from_slug(slug: str) -> str:
    # Preserve inner casing for better readability (e.g. HuggingFaceH4 -> HuggingFaceH4).
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[_\\-]+", slug) if part)


__all__ = [
    "parse_support_config",
    "ensure_git_clean",
    "guard_commands",
    "iter_test_commands",
    "detect_state",
    "show_status",
    "class_name_from_slug",
    "slugify_dataset_name",
    "artifact_slug_from_dataset_id",
    "normalize_preprocess_name",
]
