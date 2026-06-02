from __future__ import annotations

from typing import Any

from gage_eval.reporting.rendering._context import normalize_context, summarize_mapping


class MarkdownRenderer:
    """Renders a deterministic Markdown report from a report context."""

    def render(self, context: dict[str, Any]) -> str:
        payload = normalize_context(context)
        lines: list[str] = ["# GAGE Run Report", ""]
        self._headline(lines, payload)
        self._mapping_section(lines, "Runtime Health", payload.get("runtime_health"))
        self._mapping_section(lines, "Observability Health", payload.get("observability_health"))
        self._list_section(lines, "Metrics", payload.get("metrics"), empty="No metrics recorded.")
        self._list_section(lines, "Attention Cases", payload.get("attention_cases"), id_key="case_id")
        self._list_section(lines, "Outliers", payload.get("outliers"), id_key="metric_id")
        self._list_section(lines, "Failure Clusters", payload.get("failure_clusters"), id_key="cluster_id")
        self._case_details(lines, payload.get("case_details"))
        self._list_section(lines, "Evidence", payload.get("evidence_refs"), id_key="ref_id")
        self._mapping_section(lines, "Scenario Profiles", payload.get("scenario_profiles"))
        self._mapping_section(lines, "Methodology", payload.get("methodology"))
        self._extension_sections(lines, payload.get("summary_sections"))
        return "\n".join(lines).rstrip() + "\n"

    def _headline(self, lines: list[str], payload: dict[str, Any]) -> None:
        run = payload.get("run") or {}
        headline = payload.get("headline") or {}
        lines.extend(
            [
                "## Headline",
                "",
                f"- Run ID: {run.get('run_id', 'unknown')}",
                f"- Verdict: {headline.get('verdict', 'unknown')}",
                f"- Summary: {headline.get('one_line_summary', 'No summary available.')}",
                f"- Reason: {headline.get('verdict_reason', 'No verdict reason available.')}",
                "",
            ]
        )

    def _mapping_section(self, lines: list[str], title: str, value: Any) -> None:
        lines.extend([f"## {title}", ""])
        if not isinstance(value, dict) or not value:
            lines.extend(["No data available.", ""])
            return
        for key, child in sorted(value.items()):
            lines.append(f"- {key}: {summarize_mapping(child)}")
        lines.append("")

    def _list_section(
        self,
        lines: list[str],
        title: str,
        value: Any,
        *,
        id_key: str = "metric_id",
        empty: str = "No entries recorded.",
    ) -> None:
        lines.extend([f"## {title}", ""])
        if not isinstance(value, list) or not value:
            lines.extend([empty, ""])
            return
        for index, item in enumerate(value, start=1):
            if isinstance(item, dict):
                label = item.get(id_key) or item.get("title") or item.get("kind") or f"item-{index}"
                lines.append(f"- {label}: {summarize_mapping(item)}")
            else:
                lines.append(f"- {item}")
        lines.append("")

    def _case_details(self, lines: list[str], value: Any) -> None:
        lines.extend(["## Case Details", ""])
        if not isinstance(value, dict) or not value:
            lines.extend(["No case details recorded.", ""])
            return
        for key, child in sorted(value.items()):
            lines.append(f"- {key}: {summarize_mapping(child)}")
        lines.append("")

    def _extension_sections(self, lines: list[str], sections: Any) -> None:
        lines.extend(["## Extension Sections", ""])
        if not isinstance(sections, list) or not sections:
            lines.extend(["No extension sections recorded.", ""])
            return
        for section in sections:
            if not isinstance(section, dict):
                lines.append(f"- {section}")
                continue
            title = section.get("title") or section.get("section_id") or "section"
            lines.append(f"- {title}: {summarize_mapping(section)}")
        lines.append("")
