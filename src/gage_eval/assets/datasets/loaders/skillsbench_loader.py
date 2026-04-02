"""SkillsBench dataset loader backed by Harbor task directories."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Iterator, Optional

from gage_eval.assets.datasets.hubs.base import DatasetHubHandle
from gage_eval.assets.datasets.loaders.base import DatasetLoader
from gage_eval.assets.datasets.loaders.loader_utils import apply_default_params, apply_preprocess
from gage_eval.assets.datasets.manager import DataSource
from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.registry import registry

try:  # pragma: no cover - exercised on the interpreter actually used in CI
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

_DEFAULT_REPO_URL = "https://github.com/laude-institute/harbor-datasets.git"
_DEFAULT_REVISION = "main"
_DEFAULT_SUBDIR = "datasets/skillsbench"
_ENV_ASSIGN_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_VAR_PATTERN = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))")
_ENV_PLACEHOLDER_PATTERN = re.compile(r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}$")
_HOST_CODEX_HOME_ENV = "GAGE_CODEX_HOST_HOME"
_CONTAINER_CODEX_HOST_HOME = "/gage-host-codex"


@registry.asset(
    "dataset_loaders",
    "skillsbench_harbor",
    desc="SkillsBench loader backed by Harbor datasets checkout",
    tags=("skillsbench", "harbor", "docker"),
    supports_streaming=False,
)
class SkillsBenchHarborLoader(DatasetLoader):
    """Materialize SkillsBench task directories into GAGE samples."""

    def load(self, hub_handle: Optional[DatasetHubHandle], *, trace=None) -> DataSource:
        params = dict(self.spec.params or {})
        dataset_root = _resolve_dataset_root(self.spec, params)
        records = list(_iter_skillbench_samples(dataset_root, params))
        records = apply_preprocess(
            records,
            self.spec,
            data_path=str(dataset_root),
            registry_lookup=self.registry_lookup,
            allow_lazy_import=self.allow_asset_lazy_import,
            doc_to_text=None,
            doc_to_visual=None,
            doc_to_audio=None,
            trace=trace,
        )
        records = list(apply_default_params(records, self.spec))
        return DataSource(
            dataset_id=self.spec.dataset_id,
            records=records,
            metadata={
                "loader": "skillsbench_harbor",
                "repo_url": params.get("repo_url", _DEFAULT_REPO_URL),
                "revision": params.get("revision", _DEFAULT_REVISION),
                "dataset_root": str(dataset_root),
                "sample_count": len(records),
            },
            validation=self.spec.schema,
            streaming=False,
        )


def _resolve_dataset_root(spec: DatasetSpec, params: Dict[str, Any]) -> Path:
    local_repo_dir = params.get("local_repo_dir") or _default_local_repo_dir()
    revision = str(params.get("revision") or _DEFAULT_REVISION)
    repo_url = str(params.get("repo_url") or _DEFAULT_REPO_URL)
    dataset_subdir = str(params.get("dataset_subdir") or _DEFAULT_SUBDIR)
    auto_download = bool(params.get("auto_download", True))
    update_existing = bool(params.get("update_existing", False))

    repo_root = Path(local_repo_dir).expanduser().resolve()
    target_root = repo_root / dataset_subdir
    if target_root.exists():
        if update_existing and (repo_root / ".git").exists():
            _update_checkout(repo_root, revision)
        return target_root
    if not auto_download:
        raise FileNotFoundError(
            f"SkillsBench dataset root not found: {target_root}. "
            "Set `auto_download: true` or provide an existing `local_repo_dir`."
        )
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    if not repo_root.exists():
        _clone_sparse_checkout(repo_root, repo_url=repo_url, revision=revision, dataset_subdir=dataset_subdir)
    else:
        _ensure_sparse_checkout(repo_root, dataset_subdir)
        _update_checkout(repo_root, revision)
    if not target_root.exists():
        raise FileNotFoundError(f"SkillsBench dataset root missing after sync: {target_root}")
    return target_root


def _clone_sparse_checkout(repo_root: Path, *, repo_url: str, revision: str, dataset_subdir: str) -> None:
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--branch",
            revision,
            repo_url,
            str(repo_root),
        ],
        check=True,
    )
    _ensure_sparse_checkout(repo_root, dataset_subdir)


def _ensure_sparse_checkout(repo_root: Path, dataset_subdir: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo_root), "sparse-checkout", "set", dataset_subdir],
        check=True,
    )


def _update_checkout(repo_root: Path, revision: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo_root), "fetch", "--depth", "1", "origin", revision],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "checkout", "FETCH_HEAD"],
        check=True,
    )


def _iter_skillbench_samples(dataset_root: Path, params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    include = {str(item) for item in params.get("include_tasks", []) or []}
    exclude = {str(item) for item in params.get("exclude_tasks", []) or []}
    skip_missing_env_tasks = bool(
        params.get("skip_missing_env_tasks")
        or params.get("skip_tasks_requiring_env")
    )
    limit = _coerce_limit(params.get("limit"))
    min_build_timeout_sec = _coerce_timeout_sec(params.get("min_build_timeout_sec"))
    build_retry_attempts = _coerce_retry_attempts(params.get("build_retry_attempts"))
    build_retry_backoff_s = _coerce_float(params.get("build_retry_backoff_s"))
    generated_dir = Path(params.get("generated_dir") or _default_generated_dir()).expanduser().resolve()
    codex_base_image = str(params.get("codex_base_image") or "gage-codex-sandbox:latest")

    count = 0
    for task_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        task_name = task_dir.name
        if include and task_name not in include:
            continue
        if task_name in exclude:
            continue
        sample = _build_sample(
            task_dir,
            generated_dir=generated_dir,
            codex_base_image=codex_base_image,
            min_build_timeout_sec=min_build_timeout_sec,
            build_retry_attempts=build_retry_attempts,
            build_retry_backoff_s=build_retry_backoff_s,
        )
        skillsbench_meta = sample.get("metadata", {}).get("skillsbench", {})
        missing_env_vars = skillsbench_meta.get("missing_env_vars") if isinstance(skillsbench_meta, dict) else []
        if skip_missing_env_tasks and missing_env_vars:
            continue
        yield sample
        count += 1
        if limit is not None and count >= limit:
            break


def _build_sample(
    task_dir: Path,
    *,
    generated_dir: Path,
    codex_base_image: str,
    min_build_timeout_sec: Optional[int],
    build_retry_attempts: int,
    build_retry_backoff_s: float,
) -> Dict[str, Any]:
    instruction_path = task_dir / "instruction.md"
    task_toml_path = task_dir / "task.toml"
    task_payload = tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
    metadata = dict(task_payload.get("metadata") or {})
    environment_cfg = dict(task_payload.get("environment") or {})
    agent_cfg = dict(task_payload.get("agent") or {})
    verifier_cfg = dict(task_payload.get("verifier") or {})
    solution_cfg = dict(task_payload.get("solution") or {})
    environment_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    solution_dir = task_dir / "solution"
    dockerfile_path = environment_dir / "Dockerfile"
    workdir = _resolve_workdir(dockerfile_path)
    runtime_env, runtime_env_sources, runtime_missing_env_vars = _resolve_env_mapping(
        dict(environment_cfg.get("env") or {}),
        dict(agent_cfg.get("env") or {}),
        dict(solution_cfg.get("env") or {}),
    )
    verifier_env, verifier_env_sources, verifier_missing_env_vars = _resolve_env_mapping(
        dict(environment_cfg.get("env") or {}),
        dict(solution_cfg.get("env") or {}),
        dict(verifier_cfg.get("env") or {}),
    )
    runtime_volumes = _resolve_runtime_volumes()
    if runtime_volumes:
        runtime_env.setdefault(_HOST_CODEX_HOME_ENV, _CONTAINER_CODEX_HOST_HOME)
    generated_dockerfile = _ensure_gage_dockerfile(
        source_dockerfile=dockerfile_path,
        task_name=task_dir.name,
        generated_dir=generated_dir,
        codex_base_image=codex_base_image,
    )
    image_tag = _resolve_image_tag(task_dir.name, dockerfile_path, codex_base_image)
    resources = _build_resource_payload(environment_cfg)
    task_build_timeout_sec = _coerce_timeout_sec(environment_cfg.get("build_timeout_sec")) or 1800
    effective_build_timeout_sec = max(task_build_timeout_sec, min_build_timeout_sec or 0)
    sample_metadata: Dict[str, Any] = {
        "benchmark_kit_id": "skillsbench",
        "task_id": task_dir.name,
        "workspace_root": workdir,
        "timeout_sec": int(agent_cfg.get("timeout_sec") or 1800),
        "skillsbench": {
            "task_id": task_dir.name,
            "task_root": str(task_dir),
            "environment_dir": str(environment_dir),
            "tests_dir": str(tests_dir),
            "solution_dir": str(solution_dir),
            "task_toml": str(task_toml_path),
            "instruction_path": str(instruction_path),
            "generated_dockerfile": str(generated_dockerfile),
            "dockerfile": str(dockerfile_path),
            "image": image_tag,
            "workdir": workdir,
            "difficulty": metadata.get("difficulty"),
            "category": metadata.get("category"),
            "tags": list(metadata.get("tags") or []),
            "required_skills": list(metadata.get("required_skills") or []),
            "agent_timeout_sec": int(agent_cfg.get("timeout_sec") or 1800),
            "verifier_timeout_sec": int(verifier_cfg.get("timeout_sec") or 1800),
            "task_build_timeout_sec": task_build_timeout_sec,
            "build_timeout_sec": effective_build_timeout_sec,
            "build_retry_attempts": build_retry_attempts,
            "build_retry_backoff_s": build_retry_backoff_s,
            "cpus": environment_cfg.get("cpus"),
            "memory_mb": environment_cfg.get("memory_mb"),
            "memory": environment_cfg.get("memory"),
            "allow_internet": environment_cfg.get("allow_internet"),
            "resource_limits": dict(resources),
            "runtime_env": dict(runtime_env),
            "runtime_env_sources": dict(runtime_env_sources),
            "verifier_env": dict(verifier_env),
            "verifier_env_sources": dict(verifier_env_sources),
            "missing_env_vars": sorted(set(runtime_missing_env_vars) | set(verifier_missing_env_vars)),
        },
    }
    instruction = instruction_path.read_text(encoding="utf-8").strip()
    return {
        "id": task_dir.name,
        "sample_id": task_dir.name,
        "instance_id": task_dir.name,
        "instruction": instruction,
        "cwd": workdir,
        "workspace_root": workdir,
        "messages": [{"role": "user", "content": [{"type": "text", "text": instruction}]}],
        "metadata": sample_metadata,
        "resources": resources,
        "sandbox": {
            "runtime_configs": {
                "build_context": str(environment_dir),
                "dockerfile": str(generated_dockerfile),
                "image": image_tag,
                "container_name_prefix": f"gage-skillsbench-{task_dir.name}",
                "command": ["sleep", "infinity"],
                "workdir": workdir,
                "exec_workdir": workdir,
                "env": dict(runtime_env),
                "volumes": list(runtime_volumes),
                "wait_for_ready": False,
                "build_timeout_s": effective_build_timeout_sec,
                "build_retry_attempts": build_retry_attempts,
                "build_retry_backoff_s": build_retry_backoff_s,
                "cpus": resources.get("cpu"),
                "memory": resources.get("memory"),
            }
        },
    }


def _build_resource_payload(environment_cfg: Dict[str, Any]) -> Dict[str, Any]:
    resources: Dict[str, Any] = {}
    cpus = environment_cfg.get("cpus")
    memory_mb = environment_cfg.get("memory_mb")
    memory = environment_cfg.get("memory")
    if cpus is not None:
        resources["cpu"] = cpus
    if memory is not None:
        resources["memory"] = str(memory)
    elif memory_mb is not None:
        resources["memory"] = f"{memory_mb}m"
    return resources


def _resolve_runtime_volumes() -> list[str]:
    host_codex_home = _resolve_host_codex_home()
    if host_codex_home is None:
        return []
    return [f"{host_codex_home}:{_CONTAINER_CODEX_HOST_HOME}:ro"]


def _resolve_host_codex_home() -> Optional[str]:
    candidate = os.environ.get(_HOST_CODEX_HOME_ENV)
    if candidate:
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return str(path)
        return None
    default = Path.home() / ".codex"
    if default.exists():
        return str(default.resolve())
    return None


def _resolve_workdir(dockerfile_path: Path) -> str:
    text = dockerfile_path.read_text(encoding="utf-8", errors="replace")
    envs: Dict[str, str] = {}
    workdir = "/app"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("ENV "):
            envs.update(_parse_env_line(line[4:].strip()))
            continue
        if line.startswith("WORKDIR "):
            workdir = _substitute_env_vars(line[8:].strip(), envs)
    return workdir


def _parse_env_line(payload: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for token in payload.split():
        match = _ENV_ASSIGN_PATTERN.match(token)
        if match:
            result[match.group(1)] = match.group(2).strip('"').strip("'")
    return result


def _substitute_env_vars(value: str, envs: Dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group("braced") or match.group("plain") or ""
        return envs.get(key, match.group(0))

    return _VAR_PATTERN.sub(repl, value)


def _resolve_env_mapping(*mappings: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, str], list[str]]:
    resolved: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    missing: list[str] = []
    for mapping in mappings:
        for key, value in mapping.items():
            env_key = str(key)
            source_name = _resolve_env_source_name(value)
            if source_name is None:
                resolved[env_key] = value
                continue
            sources[env_key] = source_name
            resolved[env_key] = f"${{{source_name}}}"
            if os.environ.get(source_name) is None:
                missing.append(source_name)
    return resolved, sources, missing


def _resolve_env_source_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    match = _ENV_PLACEHOLDER_PATTERN.fullmatch(text)
    if match:
        return match.group("name")
    return None


def _ensure_gage_dockerfile(
    *,
    source_dockerfile: Path,
    task_name: str,
    generated_dir: Path,
    codex_base_image: str,
) -> Path:
    generated_task_dir = generated_dir / task_name
    generated_task_dir.mkdir(parents=True, exist_ok=True)
    target = generated_task_dir / "Dockerfile.gage"
    payload = _render_gage_dockerfile(
        source_dockerfile.read_text(encoding="utf-8", errors="replace"),
        codex_base_image=codex_base_image,
    )
    if not target.exists() or target.read_text(encoding="utf-8", errors="replace") != payload:
        target.write_text(payload, encoding="utf-8")
    return target


def _render_gage_dockerfile(source: str, *, codex_base_image: str) -> str:
    suffix = f"""

# GAGE: inject Codex runtime from the shared base image.
ENV CODEX_HOME=/root/.codex
COPY --from={codex_base_image} /usr/local/bin/node /usr/local/bin/node
COPY --from={codex_base_image} /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf ../lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex
ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
"""
    return source.rstrip() + "\n" + suffix.lstrip()


def _resolve_image_tag(task_name: str, dockerfile_path: Path, codex_base_image: str) -> str:
    rendered = _render_gage_dockerfile(
        dockerfile_path.read_text(encoding="utf-8", errors="replace"),
        codex_base_image=codex_base_image,
    )
    digest = hashlib.sha1(
        (task_name + "\n" + rendered).encode("utf-8")
    ).hexdigest()[:12]
    return f"gage-skillsbench-{task_name}:{digest}"


def _default_local_repo_dir() -> Path:
    return _repo_root().parent / "data" / "local-deps" / "harbor-datasets"


def _default_generated_dir() -> Path:
    return _repo_root().parent / "data" / "local-deps" / "skillsbench_runtime"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _coerce_limit(value: Any) -> Optional[int]:
    if value in (None, "", 0):
        return None
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return None


def _coerce_timeout_sec(value: Any) -> Optional[int]:
    if value in (None, "", 0):
        return None
    try:
        return max(1, int(float(value)))
    except (TypeError, ValueError):
        return None


def _coerce_retry_attempts(value: Any) -> int:
    if value in (None, "", 0):
        return 2
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 2


def _coerce_float(value: Any) -> float:
    if value in (None, ""):
        return 3.0
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 3.0


__all__ = ["SkillsBenchHarborLoader"]
