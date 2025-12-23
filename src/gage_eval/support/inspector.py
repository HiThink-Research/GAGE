from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from loguru import logger

from .config import SupportConfig
from .utils import slugify_dataset_name


def _read_local_records(path: Path, max_samples: int) -> Tuple[List[Dict[str, Any]], int]:
    if path.is_dir():
        jsonl_files = sorted(path.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No jsonl found under {path}")
        path = jsonl_files[0]
    records: List[Dict[str, Any]] = []
    total = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            total += 1
            if len(records) < max_samples:
                records.append(json.loads(line))
    return records, total


def _read_hf_records(hub_id: str, subset: Optional[str], split: Optional[str], max_samples: int) -> Tuple[List[Dict[str, Any]], int]:
    from datasets import load_dataset  # local import to avoid heavy import at CLI boot

    ds = load_dataset(hub_id, subset, split=split or "train", streaming=False, trust_remote_code=False)
    # datasets may return DatasetDict or Dataset
    if isinstance(ds, dict):
        if split and split in ds:
            ds = ds[split]
        else:
            ds = next(iter(ds.values()))
    size = len(ds)
    records = [dict(ds[i]) for i in range(min(max_samples, size))]
    return records, size


def _collect_field_paths(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            yield path, v
            yield from _collect_field_paths(v, path)
    elif isinstance(obj, list):
        # do not include indices in path, only infer element types
        for v in obj:
            yield from _collect_field_paths(v, prefix)


def infer_schema(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Infer a lightweight schema summary from sample records."""

    total = max(len(records), 1)
    field_types: Dict[str, set[str]] = {}
    presence: Dict[str, int] = {}

    for rec in records:
        seen_paths = set()
        for path, value in _collect_field_paths(rec):
            seen_paths.add(path)
            tname = type(value).__name__
            field_types.setdefault(path, set()).add(tname)
        for p in seen_paths:
            presence[p] = presence.get(p, 0) + 1

    missing_rate = {p: 1.0 - presence.get(p, 0) / total for p in field_types.keys()}

    modalities = sorted(_detect_modalities(records))
    fields = {p: {"types": sorted(list(ts)), "missing_rate": missing_rate[p]} for p, ts in field_types.items()}
    return {"fields": fields, "modalities": modalities, "record_count": len(records)}


def _detect_modalities(records: List[Dict[str, Any]]) -> set[str]:
    modalities: set[str] = set()

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            t = x.get("type")
            if isinstance(t, str):
                if t in ("text", "text_url"):
                    modalities.add("text")
                if "image" in t:
                    modalities.add("image")
                if "audio" in t:
                    modalities.add("audio")
                if "video" in t:
                    modalities.add("video")
                if "file" in t:
                    modalities.add("file")
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for rec in records:
        walk(rec)
    if not modalities:
        modalities.add("text")
    return modalities


def inspect_dataset(
    *,
    dataset_name: str,
    subset: Optional[str],
    split: Optional[str],
    max_samples: int,
    local_path: Optional[Path],
    cfg: SupportConfig,
) -> Path:
    """Inspect a dataset and write meta/sample/schema into workspace."""

    workspace_root = cfg.paths.workspace_root
    slug_source = dataset_name if not local_path else local_path.name
    slug = slugify_dataset_name(slug_source)
    dataset_dir = workspace_root / slug
    dataset_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]]
    size: int
    source: Union[str, Path]
    if local_path or Path(dataset_name).exists():
        source = local_path or Path(dataset_name)
        records, size = _read_local_records(Path(source), max_samples)
    else:
        source = dataset_name
        try:
            records, size = _read_hf_records(dataset_name, subset, split, max_samples)
        except Exception as exc:
            logger.error(f"Failed to load HF dataset {dataset_name}: {exc}")
            raise

    meta = {
        "hub_id": str(source),
        "subset": subset,
        "split": split,
        "size": size,
        "slug": slug,
    }
    (dataset_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (dataset_dir / "sample.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    schema = infer_schema(records)
    (dataset_dir / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"Inspected dataset '{dataset_name}' -> {dataset_dir}")
    return dataset_dir


__all__ = ["inspect_dataset", "infer_schema"]
