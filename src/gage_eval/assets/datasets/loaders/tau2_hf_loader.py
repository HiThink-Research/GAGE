"""Tau2 task loader (HuggingFace Hub + local fallback)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from gage_eval.assets.datasets.hubs.base import DatasetHubHandle
from gage_eval.assets.datasets.loaders.base import DatasetLoader
from gage_eval.assets.datasets.loaders.loader_utils import apply_default_params, apply_preprocess
from gage_eval.assets.datasets.manager import DataSource
from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.registry import registry
from gage_eval.utils.benchmark_helpers.tau2 import (
    ensure_tau2_importable,
    load_tau2_split,
    load_tau2_tasks,
    resolve_tau2_data_dir,
)


_TASKS_FILENAME = "tasks.json"
_SPLIT_PREFIX = "split_"
_DEFAULT_TASK_SPLIT = "base"
_DOMAIN_TASKSET_FALLBACK = {"telecom-workflow": "telecom"}


@registry.asset(
    "dataset_loaders",
    "tau2_tasks",
    desc="Tau2 tasks loader (HF Hub + local data)",
    tags=("tau2", "huggingface", "local"),
    supports_streaming=False,
)
class Tau2TasksLoader(DatasetLoader):
    """Load Tau2 tasks from HuggingFace Hub or a local data directory."""

    def load(self, hub_handle: Optional[DatasetHubHandle], *, trace=None) -> DataSource:
        params = _merge_params(self.spec, hub_handle)
        domain = _coerce_str(params.get("domain") or params.get("task_domain"))
        if not domain:
            raise ValueError(f"Dataset '{self.spec.dataset_id}' requires 'domain' for tau2_tasks loader")

        task_set = _coerce_str(params.get("task_set") or params.get("task_set_name"))
        if not task_set:
            task_set = _DOMAIN_TASKSET_FALLBACK.get(domain, domain)

        hub_id = _coerce_str(params.get("hub_id") or params.get("path") or params.get("hf_repo_id"))
        source = _normalize_source(params.get("source") or ("huggingface" if hub_id else "local"))
        snapshot_path: Optional[Path] = None

        if source != "local":
            if source != "huggingface":
                raise ValueError(f"tau2_tasks loader source '{source}' is unsupported")
            if not hub_id:
                raise ValueError(f"Dataset '{self.spec.dataset_id}' missing 'hub_id' for tau2_tasks loader")
            snapshot_path = _download_hf_snapshot(hub_id, params)
            data_dir = _resolve_hf_data_dir(params, snapshot_path)
        else:
            data_dir = resolve_tau2_data_dir(
                params.get("data_dir") or params.get("tau2_data_dir") or params.get("data_path")
            )

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Tau2 data directory not found: {data_dir}. "
                "Set TAU2_DATA_DIR or run 'tau2 check-data' to validate your data path."
            )

        tasks_path = _resolve_tasks_path(params, data_dir, task_set)
        split_path = _resolve_split_path(params, tasks_path)
        task_split = _resolve_task_split(params)
        allow_non_base = bool(params.get("allow_non_base_split") or params.get("allow_non_base"))
        if task_split != _DEFAULT_TASK_SPLIT and not allow_non_base:
            logger.warning(
                "Tau2 loader using non-base split '{}' (dataset='{}'). "
                "Set allow_non_base_split=true to silence this warning.",
                task_split,
                self.spec.dataset_id,
            )

        tasks = load_tau2_tasks(tasks_path)
        split_mapping = load_tau2_split(split_path) if split_path and split_path.exists() else None
        tasks = _filter_by_split(tasks, split_mapping, task_split, tasks_path, split_path)
        task_ids = _coerce_list(params.get("task_ids"))
        if task_ids:
            allowed = {str(item) for item in task_ids}
            tasks = [task for task in tasks if str(task.get("id")) in allowed]

        if params.get("shuffle"):
            seed = _coerce_int(params.get("seed"), default=300)
            random.Random(seed).shuffle(tasks)
        num_tasks = _coerce_int(params.get("num_tasks"))
        if num_tasks is not None:
            tasks = tasks[: max(0, num_tasks)]

        if str(params.get("preprocess") or "").strip() == "tau2_preprocessor":
            ensure_tau2_importable()

        num_trials = max(1, _coerce_int(params.get("num_trials"), default=1))
        base_seed = params.get("seed")
        records: List[Dict[str, Any]] = []
        for task in tasks:
            for trial in range(num_trials):
                record = dict(task)
                record["trial"] = trial
                record["seed"] = (int(base_seed) + trial) if base_seed is not None else None
                record["_tau2_domain"] = domain
                record["_tau2_task_set"] = task_set
                record["_tau2_split"] = task_split
                records.append(record)

        records = apply_preprocess(
            records,
            self.spec,
            data_path=str(tasks_path),
            trace=trace,
        )
        records = apply_default_params(records, self.spec)

        metadata = {
            "loader": "tau2_tasks",
            "domain": domain,
            "task_set": task_set,
            "task_split": task_split,
            "num_trials": num_trials,
            "data_dir": str(data_dir),
            "tasks_path": str(tasks_path),
            "split_path": str(split_path) if split_path else None,
            "hub_id": hub_id,
            "source": source,
            "snapshot_path": str(snapshot_path) if snapshot_path else None,
            "revision": params.get("revision"),
        }
        return DataSource(
            dataset_id=self.spec.dataset_id,
            records=list(records),
            metadata=metadata,
            validation=self.spec.schema,
            streaming=False,
        )


def _merge_params(spec: DatasetSpec, hub_handle: Optional[DatasetHubHandle]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if hub_handle and isinstance(hub_handle.metadata, dict):
        merged.update(hub_handle.metadata)
    merged.update(spec.params or {})
    return merged


def _resolve_task_split(params: Dict[str, Any]) -> str:
    split = params.get("task_split") or params.get("task_split_name") or params.get("split")
    if split is None or str(split).strip() == "":
        return _DEFAULT_TASK_SPLIT
    return str(split)


def _resolve_tasks_path(params: Dict[str, Any], data_dir: Path, task_set: str) -> Path:
    explicit = params.get("tasks_path") or params.get("tasks_file")
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = [
        data_dir / "tau2" / "domains" / task_set / _TASKS_FILENAME,
        data_dir / "domains" / task_set / _TASKS_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_split_path(params: Dict[str, Any], tasks_path: Path) -> Optional[Path]:
    explicit = params.get("split_path") or params.get("split_file")
    if explicit:
        return Path(explicit).expanduser().resolve()
    stem = tasks_path.stem
    return tasks_path.parent / f"{_SPLIT_PREFIX}{stem}.json"


def _filter_by_split(
    tasks: List[Dict[str, Any]],
    split_mapping: Optional[Dict[str, List[str]]],
    task_split: str,
    tasks_path: Path,
    split_path: Optional[Path],
) -> List[Dict[str, Any]]:
    if not task_split:
        return tasks
    if not split_mapping:
        if task_split != _DEFAULT_TASK_SPLIT:
            raise ValueError(
                f"Tau2 split '{task_split}' requested but split file not found: {split_path or tasks_path.parent}"
            )
        return tasks
    if task_split not in split_mapping:
        raise ValueError(
            f"Tau2 split '{task_split}' not found in {split_path}. "
            f"Available splits: {sorted(split_mapping.keys())}"
        )
    allowed = {str(task_id) for task_id in split_mapping[task_split]}
    return [task for task in tasks if str(task.get("id")) in allowed]


def _download_hf_snapshot(hub_id: str, params: Dict[str, Any]) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("`huggingface_hub` package is required to load tau2 datasets from HF") from exc

    repo_type = _coerce_str(params.get("repo_type")) or "dataset"
    cache_dir = _coerce_str(params.get("cache_dir"))
    local_dir = _coerce_str(params.get("download_dir") or params.get("local_dir"))
    if not local_dir:
        local_dir = _coerce_str(
            params.get("data_dir") or params.get("tau2_data_dir") or params.get("data_path")
        )
    revision = _coerce_str(params.get("revision"))
    token = _coerce_str(params.get("token"))
    allow_patterns = _coerce_patterns(params.get("allow_patterns"))
    ignore_patterns = _coerce_patterns(params.get("ignore_patterns"))
    local_files_only = _coerce_bool(params.get("local_files_only"))
    force_download = _coerce_bool(params.get("force_download"))
    local_dir_use_symlinks = params.get("local_dir_use_symlinks")

    logger.info(
        "Downloading tau2 dataset from HF hub_id={} revision={} cache_dir={} local_dir={} ",
        hub_id,
        revision or "latest",
        cache_dir or "auto",
        local_dir or "auto",
    )

    download_kwargs: Dict[str, Any] = {}
    if cache_dir:
        download_kwargs["cache_dir"] = cache_dir
    if local_dir:
        download_kwargs["local_dir"] = local_dir
    if revision:
        download_kwargs["revision"] = revision
    if token:
        download_kwargs["token"] = token
    if allow_patterns:
        download_kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns:
        download_kwargs["ignore_patterns"] = ignore_patterns
    if local_files_only is not None:
        download_kwargs["local_files_only"] = local_files_only
    if force_download is not None:
        download_kwargs["force_download"] = force_download
    if local_dir_use_symlinks is not None:
        download_kwargs["local_dir_use_symlinks"] = local_dir_use_symlinks

    snapshot_path = snapshot_download(repo_id=hub_id, repo_type=repo_type, **download_kwargs)
    return Path(snapshot_path).expanduser().resolve()


def _resolve_hf_data_dir(params: Dict[str, Any], snapshot_path: Path) -> Path:
    explicit = _coerce_str(params.get("data_dir") or params.get("tau2_data_dir") or params.get("data_path"))
    if explicit:
        explicit_path = Path(explicit).expanduser().resolve()
        if _has_tau2_domains(explicit_path):
            return explicit_path
    data_subdir = _coerce_str(params.get("data_subdir") or params.get("hf_data_subdir"))
    candidates: List[Path] = []
    if data_subdir:
        candidates.append(snapshot_path / data_subdir)
    candidates.extend([snapshot_path, snapshot_path / "data"])
    for candidate in candidates:
        if _has_tau2_domains(candidate):
            return candidate
    raise FileNotFoundError(
        "Tau2 HF snapshot missing expected task directories. "
        "Set data_dir or data_subdir to the directory containing 'tau2/domains' or 'domains'."
    )


def _has_tau2_domains(path: Path) -> bool:
    return (path / "tau2" / "domains").exists() or (path / "domains").exists()


def _normalize_source(value: Optional[str]) -> str:
    if value is None:
        return "local"
    normalized = str(value).strip().lower()
    if normalized in {"local", "disk"}:
        return "local"
    if normalized in {"hf", "huggingface"}:
        return "huggingface"
    return normalized


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value)]


def _coerce_patterns(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value)]


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return None


def _coerce_int(value: Any, *, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
