"""Shared helpers for canonical arena replay artifact paths."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping


def sanitize_sample_id(value: object | None) -> str:
    """Return a filesystem-safe sample id."""

    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "")).strip("_")
    return cleaned or "unknown"


def resolve_invocation_run_sample_ids(
    *,
    invocation_context: Any | None,
    run_id: str | None = None,
    sample_id: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve run/sample ids from explicit args, invocation context, or env vars."""

    resolved_run_id = _normalize_optional_text(run_id)
    resolved_sample_id = _normalize_optional_text(sample_id)

    if invocation_context is not None:
        if resolved_run_id is None:
            trace = getattr(invocation_context, "trace", None)
            resolved_run_id = _normalize_optional_text(getattr(trace, "run_id", None))
        if resolved_sample_id is None:
            sample_payload = getattr(invocation_context, "sample_payload", None)
            if isinstance(sample_payload, Mapping):
                resolved_sample_id = _normalize_optional_text(
                    sample_payload.get("id") or sample_payload.get("sample_id")
                )

    if resolved_run_id is None:
        resolved_run_id = _normalize_optional_text(os.environ.get("GAGE_EVAL_RUN_ID"))
    if resolved_sample_id is None:
        resolved_sample_id = _normalize_optional_text(os.environ.get("GAGE_EVAL_SAMPLE_ID"))
    return resolved_run_id, resolved_sample_id


def resolve_replay_manifest_path(
    *,
    run_id: str | None,
    sample_id: str | None,
    output_dir: str | None = None,
    base_dir: str | None = None,
) -> Path | None:
    """Resolve the canonical replay manifest path.

    The canonical layout is:
    - ``runs/<run_id>/replays/<sample_id>/replay.json`` when ``output_dir`` is not provided
    - ``<output_dir>/<sample_id>/replay.json`` when ``output_dir`` overrides the replay root
    """

    resolved_sample_id = _normalize_optional_text(sample_id) or _normalize_optional_text(
        os.environ.get("GAGE_EVAL_SAMPLE_ID")
    )
    safe_sample_id = sanitize_sample_id(resolved_sample_id)
    if output_dir:
        replay_root = Path(output_dir).expanduser()
        if not replay_root.is_absolute():
            replay_root = (Path.cwd() / replay_root).resolve()
        return replay_root / safe_sample_id / "replay.json"

    resolved_run_id = _normalize_optional_text(run_id) or _normalize_optional_text(
        os.environ.get("GAGE_EVAL_RUN_ID")
    )
    if resolved_run_id is None:
        return None
    replay_base = Path(base_dir or os.environ.get("GAGE_EVAL_SAVE_DIR", "./runs")).expanduser()
    if not replay_base.is_absolute():
        replay_base = (Path.cwd() / replay_base).resolve()
    return replay_base / resolved_run_id / "replays" / safe_sample_id / "replay.json"


def _normalize_optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
