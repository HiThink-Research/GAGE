"""Sample-related helper utilities."""

from __future__ import annotations

from typing import Any, Optional


def extract_sample_id(sample: Optional[dict]) -> Optional[str]:
    """Best-effort extraction of a sample identifier."""

    if not isinstance(sample, dict):
        return None
    if "sample_id" in sample and sample["sample_id"] is not None:
        return str(sample["sample_id"])
    if "id" in sample and sample["id"] is not None:
        return str(sample["id"])
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        meta_id = metadata.get("sample_id") or metadata.get("id")
        if meta_id is not None:
            return str(meta_id)
    return None
