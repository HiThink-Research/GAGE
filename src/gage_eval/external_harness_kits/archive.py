"""Run-level raw artifact archive helpers for external harness kits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


RAW_ARCHIVE_SCHEMA_VERSION = "gage.external_harness.raw_archive.v1"


def write_raw_archive_entry(
    *,
    archive_root: Path,
    run_id: str,
    task_id: str,
    adapter_id: str,
    provider: str,
    job_name: str | None = None,
    artifacts: Mapping[str, Path | str | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Upsert one provider entry into the run-level external harness manifest."""

    archive_root.mkdir(parents=True, exist_ok=True)
    manifest_path = archive_root / "manifest.json"
    manifest = _load_manifest(manifest_path, run_id=run_id)
    entry = {
        "task_id": str(task_id),
        "adapter_id": str(adapter_id),
        "provider": str(provider),
        "job_name": str(job_name) if job_name else None,
        "artifacts": {
            key: _path_ref(value, archive_root=archive_root)
            for key, value in dict(artifacts or {}).items()
            if value is not None
        },
        "metadata": dict(metadata or {}),
    }
    entries = [
        item
        for item in manifest.get("entries", [])
        if not _same_entry(item, entry)
    ]
    entries.append(_strip_empty(entry))
    manifest["entries"] = entries
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return manifest_path


def safe_archive_segment(value: str) -> str:
    """Return a stable path segment for task and adapter ids."""

    text = str(value).strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return safe or "unnamed"


def _load_manifest(path: Path, *, run_id: str) -> dict[str, Any]:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if isinstance(existing, dict):
            existing.setdefault("schema_version", RAW_ARCHIVE_SCHEMA_VERSION)
            existing.setdefault("run_id", str(run_id))
            existing.setdefault("entries", [])
            return existing
    return {
        "schema_version": RAW_ARCHIVE_SCHEMA_VERSION,
        "run_id": str(run_id),
        "entries": [],
    }


def _same_entry(existing: Mapping[str, Any], entry: Mapping[str, Any]) -> bool:
    return (
        existing.get("task_id") == entry.get("task_id")
        and existing.get("adapter_id") == entry.get("adapter_id")
        and existing.get("provider") == entry.get("provider")
    )


def _path_ref(value: Path | str, *, archive_root: Path) -> str:
    path = Path(str(value))
    try:
        return str(path.relative_to(archive_root))
    except ValueError:
        return str(path)


def _strip_empty(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, {}, [])
    }


__all__ = [
    "RAW_ARCHIVE_SCHEMA_VERSION",
    "safe_archive_segment",
    "write_raw_archive_entry",
]
