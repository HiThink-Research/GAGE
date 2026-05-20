from __future__ import annotations

from typing import Any

from gage_eval.reporting.rendering._context import normalize_context


class PromptRenderer:
    """Renders the companion prompt for human or model-assisted report reading."""

    def render(self, context: dict[str, Any]) -> str:
        payload = normalize_context(context)
        run_id = (payload.get("run") or {}).get("run_id", "unknown")
        evidence_count = len(payload.get("evidence_refs") or [])
        return "\n".join(
            [
                "GAGE report pack reading prompt",
                "",
                f"Run ID: {run_id}",
                "Primary context file: report_context.json",
                f"EvidenceRef entries available: {evidence_count}",
                "",
                "Use report_context.json as the source of truth for the report structure.",
                "Use EvidenceRef.path values only as bounded, redacted evidence references.",
                "Do not invent metrics, cases, causes, or evidence that are not present.",
                "For safety, do not read unredacted raw trace files or bypass redacted report pack assets.",
                "If evidence is missing or degraded, state that limitation explicitly.",
                "",
            ]
        )
