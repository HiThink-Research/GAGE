from __future__ import annotations

from pathlib import Path
from typing import Any

from gage_eval.reporting.rendering._context import atomic_write_text, deterministic_json


class SummaryWriter:
    """Writes the legacy summary.json with optional report pack diagnostics."""

    def write(
        self,
        cache: Any,
        payload: dict[str, Any],
        *,
        report_pack_diagnostics: dict[str, Any] | None = None,
    ) -> Path:
        summary = dict(payload)
        if hasattr(cache, "flush_writers"):
            cache.flush_writers()
        if hasattr(cache, "buffered_writer_summary"):
            summary.update(cache.buffered_writer_summary())
        if report_pack_diagnostics is not None:
            status = report_pack_diagnostics.get("report_pack_status", "unknown")
            summary["report_pack"] = {
                "status": status,
                "diagnostics": dict(report_pack_diagnostics),
            }
        if hasattr(cache, "write_summary"):
            return cache.write_summary(summary)
        target = Path(cache.run_dir) / "summary.json"
        return atomic_write_text(target, deterministic_json(summary))
