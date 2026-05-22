from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from gage_eval.reporting.contracts import EvidenceRef
from gage_eval.reporting.rendering._context import atomic_write_text, deterministic_json, utc_now_iso


class ManifestWriter:
    """Writes report asset integrity metadata."""

    def write(
        self,
        directory: str | Path,
        *,
        run_id: str,
        evidence_refs: Iterable[EvidenceRef | dict[str, Any]] | None = None,
        files: Iterable[str | Path] | None = None,
    ) -> Path:
        base = Path(directory)
        generated_at = utc_now_iso()
        file_entries = [_file_entry(base, Path(file), generated_at) for file in (files or [])]
        evidence_entries = [
            ref.to_dict() if hasattr(ref, "to_dict") else dict(ref)
            for ref in (evidence_refs or [])
        ]
        payload = {
            "schema_version": "1.0",
            "run_id": run_id,
            "generated_at": generated_at,
            "files": file_entries,
            "evidence_refs": evidence_entries,
        }
        return atomic_write_text(base / "assets_manifest.json", deterministic_json(payload))


def _file_entry(base: Path, path: Path, generated_at: str) -> dict[str, Any]:
    full_path = path if path.is_absolute() else base / path
    data = full_path.read_bytes()
    try:
        rel_path = full_path.relative_to(base)
    except ValueError:
        rel_path = full_path
    return {
        "path": rel_path.as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "size_bytes": len(data),
        "generated_at": generated_at,
        "schema_version": "1.0",
    }
