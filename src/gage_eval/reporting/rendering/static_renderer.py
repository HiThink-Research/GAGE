from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from gage_eval.reporting.rendering._context import normalize_context


_PREVIEW_LIMIT = 1800
_DEBUG_LIMIT = 1200
_ROW_LIMIT = 8
_SVG_DATA_IMAGE_URL_RE = re.compile(
    r"data:image/svg\+xml(?:;[a-z0-9.+-]+(?:=[^\s\"'<>;,)\]}]+)?)*,"
    r"(?:.*?</svg>|[^\s\"')}\]]+)",
    re.IGNORECASE | re.DOTALL,
)
_DATA_IMAGE_URL_RE = re.compile(
    r"data:image/[a-z0-9.+-]+(?:;[a-z0-9.+-]+(?:=[^\s\"'<>;,)\]}]+)?)*,[^\s\"')}\]]+",
    re.IGNORECASE,
)


class StaticReportRenderer:
    """Renders a self-contained static HTML report without network dependencies."""

    def render(self, context: dict[str, Any]) -> str:
        payload = _sanitize_payload(normalize_context(context))
        context_json = _safe_script_json(payload)
        run = _mapping(payload.get("run"))
        title = f"GAGE Run Report - {_text(run.get('run_id'), 'unknown')}"
        main_sections = [
            _render_quick_stats(payload),
            _render_key_findings(payload),
            _render_metrics_dashboard(payload),
            _render_scenario_profile(payload),
            _render_evidence_explorer(payload),
        ]
        main_body = "\n".join(section for section in main_sections if section)
        footer = _render_footer(payload)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
  {_render_hero(payload)}
  <main>
  {main_body}
  </main>
  {footer}
  <script type="application/json" id="gage-report-context">{context_json}</script>
  <script>
  window.gage = window.gage || {{}};
  window.gage.report_context = JSON.parse(document.getElementById("gage-report-context").textContent);
  </script>
  <script data-asset-name="report.js">{_JS}</script>
</body>
</html>
"""


def _render_hero(payload: dict[str, Any]) -> str:
    run = _mapping(payload.get("run"))
    headline = _mapping(payload.get("headline"))
    verdict = _text(headline.get("verdict"), "unknown")
    summary = _text(headline.get("one_line_summary"), "Run report generated.")
    reason = _text(headline.get("verdict_reason"), "")
    metric = _primary_metric(headline, _list(payload.get("metrics")))
    metric_html = ""
    if metric:
        bar = _metric_bar(metric.get("value_raw"), metric.get("unit"))
        metric_html = f"""
        <div class="primary-metric">
          <span class="label">Primary metric</span>
          <strong>{_escape(metric["value"])}</strong>
          <span>{_escape(metric["label"])}</span>
          {bar}
        </div>"""
    return f"""<header class="report-shell hero" data-filter-target="overview">
    <div class="hero-topline">
      <span>Run report</span>
      {_badge(verdict, "verdict")}
    </div>
    <div class="hero-grid">
      <div>
        <h1>{_escape(summary)}</h1>
        <p class="muted">{_escape(reason)}</p>
      </div>
      <dl class="meta-list">
        {_definition("Run", run.get("run_id"), "unknown")}
        {_definition("Run dir", run.get("run_dir"), "")}
        {_definition("Started", run.get("started_at") or run.get("started_at_iso"), "")}
        {_definition("Finished", run.get("finished_at") or run.get("finished_at_iso"), "")}
      </dl>
      {metric_html}
    </div>
  </header>"""


def _render_quick_stats(payload: dict[str, Any]) -> str:
    run = _mapping(payload.get("run"))
    runtime = _mapping(payload.get("runtime_health"))
    observability = _mapping(payload.get("observability_health"))
    stats = [
        ("Samples", runtime.get("sample_count")),
        ("Completed", runtime.get("completed_count")),
        ("Failed", runtime.get("failed_count")),
        ("Aborted", runtime.get("aborted_count")),
        ("Duration", runtime.get("duration_s") or run.get("duration_s")),
    ]
    optional = [
        ("Events", observability.get("events_emitted_total")),
        ("Cost", _first_present(runtime, observability, "total_cost", "cost", "agent_cost")),
        ("Tokens", _first_present(runtime, observability, "total_tokens", "tokens", "token_count")),
    ]
    cards = "".join(
        _stat_card(label, value, duration=label == "Duration")
        for label, value in [*stats, *optional]
        if value is not None and value != ""
    )
    return f"""<section class="report-shell section" data-filter-target="quick-stats">
    <div class="section-heading">
      <h2>Quick stats</h2>
    </div>
    <div class="stat-grid">{cards}</div>
  </section>"""


def _render_key_findings(payload: dict[str, Any]) -> str:
    attention_cases = _list(payload.get("attention_cases"))
    failure_clusters = _list(payload.get("failure_clusters"))
    outliers = _list(payload.get("outliers"))
    if not attention_cases and not failure_clusters and not outliers:
        return ""

    blocks = []
    if attention_cases:
        blocks.append(_attention_cases_table(attention_cases, _mapping(payload.get("case_details"))))
    if failure_clusters:
        blocks.append(_failure_clusters_table(failure_clusters))
    if outliers:
        blocks.append(_outliers_table(outliers))
    return f"""<section class="report-shell section" data-filter-target="key-findings">
    <div class="section-heading">
      <h2>Key findings</h2>
    </div>
    {"".join(blocks)}
  </section>"""


def _render_metrics_dashboard(payload: dict[str, Any]) -> str:
    headline = _mapping(payload.get("headline"))
    metrics = _list(payload.get("metrics"))
    primary = _primary_metric(headline, metrics)
    primary_html = ""
    if primary:
        bar = _metric_bar(primary.get("value_raw"), primary.get("unit"))
        primary_html = f"""
      <div class="metric-focus">
        <span class="label">Primary metric</span>
        <strong>{_escape(primary["value"])}</strong>
        <span>{_escape(primary["label"])}</span>
        {bar}
      </div>"""

    rows = []
    for metric in metrics:
        item = _mapping(metric)
        rows.append(
            "<tr>"
            f"<td>{_escape(item.get('metric_id') or item.get('id') or '')}</td>"
            f"<td>{_escape(item.get('name') or item.get('label') or '')}</td>"
            f"<td>{_escape(item.get('scope') or '')}</td>"
            f"<td class=\"numeric\">{_escape(_format_scalar(item.get('value')))}</td>"
            f"<td>{_escape(item.get('unit') or '')}</td>"
            f"<td>{_escape(item.get('task_id') or item.get('section_id') or '')}</td>"
            "</tr>"
        )
    table = _table(
        ["Metric", "Name", "Scope", "Value", "Unit", "Owner"],
        rows,
        empty_message="No metrics recorded.",
    )
    return f"""<section class="report-shell section" data-filter-target="metrics">
    <div class="section-heading">
      <h2>Metrics</h2>
    </div>
    {primary_html}
    {table}
  </section>"""


def _render_scenario_profile(payload: dict[str, Any]) -> str:
    profiles = _mapping(payload.get("scenario_profiles"))
    profiles = {
        key: value
        for key, value in profiles.items()
        if _profile_has_signal(str(key), _mapping(value))
    }
    if not profiles:
        return ""

    blocks = []
    for kind in ("agent", "external_harness", "game"):
        if kind in profiles:
            blocks.append(_known_profile(kind, _mapping(profiles[kind])))
    for kind, profile in sorted(profiles.items()):
        if kind in {"agent", "external_harness", "game"}:
            continue
        blocks.append(_fallback_profile(kind, _mapping(profile)))
    return f"""<section class="report-shell section" data-filter-target="scenario-profile">
    <div class="section-heading">
      <h2>Scenario profile</h2>
    </div>
    <div class="profile-grid">{"".join(blocks)}</div>
  </section>"""


def _render_evidence_explorer(payload: dict[str, Any]) -> str:
    refs = _list(payload.get("evidence_refs"))
    if not refs:
        return ""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ref in refs:
        item = _mapping(ref)
        groups[_text(item.get("kind"), "unknown")].append(item)

    blocks = []
    for kind in sorted(groups):
        rows = []
        for ref in groups[kind][:_ROW_LIMIT]:
            preview = _preview_block(ref.get("preview"), ref.get("mime_type"))
            rows.append(
                "<tr>"
                f"<td class=\"mono\">{_escape(ref.get('ref_id') or '')}</td>"
                f"<td>{_escape(ref.get('path') or '')}{preview}</td>"
                f"<td>{_escape(_format_bytes(ref.get('size_bytes')))}</td>"
                f"<td class=\"mono\">{_escape(_short_sha(ref.get('sha256')))}</td>"
                f"<td>{_escape(ref.get('sample_id') or '')}</td>"
                f"<td>{_escape(ref.get('task_id') or '')}</td>"
                "</tr>"
            )
        blocks.append(
            f"""<details class="panel" open>
        <summary>{_escape(kind)} evidence ({len(groups[kind])})</summary>
        {_table(["Ref", "Path / preview", "Size", "SHA", "Sample", "Task"], rows)}
      </details>"""
        )
    return f"""<section class="report-shell section" data-filter-target="evidence-explorer">
    <div class="section-heading">
      <h2>Evidence explorer</h2>
    </div>
    {"".join(blocks)}
  </section>"""


def _render_footer(payload: dict[str, Any]) -> str:
    methodology = _mapping(payload.get("methodology"))
    diagnostics = _mapping(payload.get("diagnostics"))
    schema = _mapping(payload.get("schema"))
    run = _mapping(payload.get("run"))
    generated_by = _mapping(schema.get("generated_by"))
    methodology_summary = _methodology_summary(methodology)
    diagnostics_summary = _diagnostics_summary(diagnostics)
    metadata_rows = [
        ("Schema", f"{schema.get('name', '')} v{schema.get('major', '')}.{schema.get('minor', '')}"),
        ("Renderer compat", schema.get("renderer_compat")),
        ("Generated by", generated_by.get("component")),
        ("Generator version", generated_by.get("version")),
        ("Run", run.get("run_id")),
        ("Report status", diagnostics.get("report_pack_status")),
    ]
    return f"""<footer class="report-shell footer" data-filter-target="methodology">
    <details open>
      <summary>Methodology</summary>
      {methodology_summary}
    </details>
    <details>
      <summary>Diagnostics</summary>
      {diagnostics_summary}
    </details>
    <details>
      <summary>Schema and run metadata</summary>
      {_kv_table(metadata_rows)}
    </details>
  </footer>"""


def _attention_cases_table(cases: list[Any], case_details: dict[str, Any]) -> str:
    rows = []
    for case in cases[:_ROW_LIMIT]:
        item = _mapping(case)
        case_id = _text(item.get("case_id"), "")
        details = _mapping(case_details.get(case_id))
        scoring = _mapping(item.get("scoring"))
        summary = _text(item.get("summary"), "")
        detail_bits = [
            _detail_text(summary),
            _ids_line("Case details", details.get("artifact_preview_ref_ids")),
            _ids_line("Evidence refs", item.get("evidence_ref_ids") or details.get("evidence_ref_ids")),
            _ids_line("Sample/trial", [item.get("sample_id"), item.get("trial_id")]),
        ]
        if details.get("message_history_preview") or details.get("tool_call_summary"):
            detail_bits.append(_collapsible_debug("Details", details))
        rows.append(
            "<tr>"
            f"<td>{_escape(case_id)}</td>"
            f"<td>{_badge(_text(item.get('severity'), 'unknown'), 'severity')}</td>"
            f"<td>{_chips(item.get('reason_codes'))}</td>"
            f"<td>{_escape(_format_scalar(scoring.get('priority_score')))}</td>"
            f"<td>{''.join(bit for bit in detail_bits if bit)}</td>"
            "</tr>"
        )
    return f"""<div class="subsection">
      <h3>Attention cases</h3>
      {_table(["Case", "Severity", "Reason codes", "Priority", "Details"], rows)}
    </div>"""


def _failure_clusters_table(clusters: list[Any]) -> str:
    rows = []
    for cluster in clusters[:_ROW_LIMIT]:
        item = _mapping(cluster)
        details = [
            _detail_text(_text(item.get("hypothesis"), "")),
            _detail_text(_text(item.get("recommended_action"), "")),
            _ids_line("Samples", item.get("sample_ids")),
            _ids_line("Evidence refs", item.get("representative_ref_ids")),
        ]
        rows.append(
            "<tr>"
            f"<td>{_escape(item.get('label') or item.get('cluster_id') or '')}</td>"
            f"<td>{_badge(_text(item.get('severity'), 'unknown'), 'severity')}</td>"
            f"<td>{_chips(item.get('cluster_key'))}</td>"
            f"<td class=\"numeric\">{_escape(_format_scalar(item.get('count')))}</td>"
            f"<td>{''.join(bit for bit in details if bit)}</td>"
            "</tr>"
        )
    return f"""<div class="subsection">
      <h3>Failure clusters</h3>
      {_table(["Cluster", "Severity", "Reason codes", "Count", "Details"], rows)}
    </div>"""


def _outliers_table(outliers: list[Any]) -> str:
    rows = []
    for outlier in outliers[:_ROW_LIMIT]:
        item = _mapping(outlier)
        top_k = _list(item.get("top_k"))[:3]
        entries = []
        for entry in top_k:
            value = _mapping(entry)
            entries.append(
                " / ".join(
                    part
                    for part in (
                        _text(value.get("sample_id"), ""),
                        _format_scalar(value.get("value")),
                        _ids_line("refs", value.get("evidence_ref_ids"), inline=True),
                    )
                    if part
                )
            )
        rows.append(
            "<tr>"
            f"<td>{_escape(item.get('metric_id') or '')}</td>"
            f"<td>{_escape(item.get('scope') or '')}</td>"
            f"<td>{_escape(item.get('ranking') or '')}</td>"
            f"<td>{_escape('; '.join(entries))}</td>"
            "</tr>"
        )
    return f"""<div class="subsection">
      <h3>Outliers</h3>
      {_table(["Metric", "Scope", "Ranking", "Top cases"], rows)}
    </div>"""


def _known_profile(kind: str, profile: dict[str, Any]) -> str:
    titles = {
        "agent": "Agent profile",
        "external_harness": "External harness profile",
        "game": "Game profile",
    }
    field_sets = {
        "agent": [
            "sample_count",
            "trial_count",
            "completed_trial_count",
            "failed_trial_count",
            "tool_call_count",
            "representative_ref_ids",
        ],
        "external_harness": [
            "harness_name",
            "suite_count",
            "sample_count",
            "completed_count",
            "failed_count",
            "representative_ref_ids",
        ],
        "game": [
            "game_name",
            "match_count",
            "move_count",
            "illegal_move_count",
            "winner_count",
            "replay_refs",
        ],
    }
    rows = [(key.replace("_", " ").title(), profile.get(key)) for key in field_sets[kind]]
    return f"""<article class="profile-panel">
      <h3>{titles[kind]}</h3>
      {_kv_table(rows)}
    </article>"""


def _fallback_profile(kind: str, profile: dict[str, Any]) -> str:
    rows = []
    for key, value in sorted(profile.items()):
        if isinstance(value, (dict, list)):
            continue
        rows.append((key.replace("_", " ").title(), value))
        if len(rows) >= 6:
            break
    return f"""<article class="profile-panel">
      <h3>{_escape(kind)}</h3>
      {_kv_table(rows)}
    </article>"""


def _profile_has_signal(kind: str, profile: dict[str, Any]) -> bool:
    if not profile:
        return False
    if kind == "agent":
        return bool(
            _positive(profile.get("trial_count"))
            or _positive(profile.get("sample_count"))
            or _positive(profile.get("failed_trial_count"))
            or profile.get("representative_ref_ids")
            or _positive(profile.get("tool_call_count"))
        )
    if kind == "external_harness":
        return bool(
            _positive(profile.get("trial_count"))
            or _positive(profile.get("sample_count"))
            or _positive(profile.get("failed_count"))
            or _positive(profile.get("suite_count"))
            or profile.get("harness_name")
            or profile.get("harnesses")
            or profile.get("trial_rollup")
        )
    if kind == "game":
        illegal = _mapping(profile.get("illegal_actions"))
        return bool(
            profile.get("game_kits")
            or _positive(profile.get("move_count"))
            or _positive(profile.get("illegal_move_count"))
            or profile.get("replay_refs")
            or _positive(illegal.get("games"))
            or _positive(illegal.get("total"))
        )
    return _generic_profile_has_signal(profile)


def _generic_profile_has_signal(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key != "profile_version" and _generic_profile_has_signal(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_generic_profile_has_signal(item) for item in value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return value not in (None, "")


def _preview_block(preview: Any, mime_type: Any) -> str:
    if preview in (None, {}, []):
        return ""
    text = _preview_text(preview, _text(mime_type, ""))
    if not text:
        return ""
    clipped, was_truncated = _truncate(text, _PREVIEW_LIMIT)
    note = " <span class=\"muted\">truncated</span>" if was_truncated else ""
    return f"""<details class="preview">
        <summary>Preview{note}</summary>
        <pre>{_escape(clipped)}</pre>
      </details>"""


def _preview_text(preview: Any, mime_type: str) -> str:
    if isinstance(preview, str):
        return _redact_data_urls(preview)
    if isinstance(preview, dict):
        for key in ("text", "content", "json", "sample", "preview"):
            value = preview.get(key)
            if isinstance(value, str):
                return _redact_data_urls(value)
            if isinstance(value, (dict, list)):
                return json.dumps(_sanitize_payload(value), ensure_ascii=False, indent=2, sort_keys=True)
        if mime_type.startswith("image/"):
            for key in ("source", "url", "image_url"):
                value = preview.get(key)
                if isinstance(value, str):
                    return _safe_preview_source(value)
        if "json" in mime_type or "text" in mime_type:
            return json.dumps(_sanitize_payload(preview), ensure_ascii=False, indent=2, sort_keys=True)
    if isinstance(preview, list) and ("json" in mime_type or "text" in mime_type):
        return json.dumps(_sanitize_payload(preview), ensure_ascii=False, indent=2, sort_keys=True)
    return ""


def _methodology_summary(methodology: dict[str, Any]) -> str:
    if not methodology:
        return "<p class=\"muted\">Methodology metadata was not provided.</p>"
    rows = []
    for key, value in sorted(methodology.items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            rows.append((key.replace("_", " ").title(), value))
        elif isinstance(value, list):
            rows.append((key.replace("_", " ").title(), ", ".join(_text(item, "") for item in value[:6])))
        elif isinstance(value, dict):
            rows.append((key.replace("_", " ").title(), ", ".join(sorted(map(str, value.keys()))[:6])))
    return _kv_table(rows)


def _diagnostics_summary(diagnostics: dict[str, Any]) -> str:
    if not diagnostics:
        return "<p class=\"muted\">No diagnostics metadata was provided.</p>"
    rows = []
    for key in ("report_pack_status", "warnings", "errors", "source_files"):
        value = diagnostics.get(key)
        if isinstance(value, list):
            rows.append((key.replace("_", " ").title(), len(value)))
            if value:
                rows.append((f"{key} preview", json.dumps(value[:3], ensure_ascii=False, sort_keys=True)))
        elif isinstance(value, dict):
            rows.append((key.replace("_", " ").title(), len(value)))
        else:
            rows.append((key.replace("_", " ").title(), value))
    return _kv_table(rows)


def _primary_metric(headline: dict[str, Any], metrics: list[Any]) -> dict[str, Any] | None:
    primary = headline.get("primary_metric")
    metric = _mapping(primary)
    if metric:
        return {
            "label": _text(metric.get("name") or metric.get("label") or metric.get("metric_id"), "primary_metric"),
            "value": _metric_value(metric),
            "value_raw": metric.get("value"),
            "unit": metric.get("unit"),
        }
    if isinstance(primary, str):
        for item in metrics:
            metric = _mapping(item)
            if primary in {metric.get("metric_id"), metric.get("id"), metric.get("name")}:
                return {
                    "label": _text(metric.get("name") or metric.get("metric_id"), primary),
                    "value": _metric_value(metric),
                    "value_raw": metric.get("value"),
                    "unit": metric.get("unit"),
                }
        return {"label": primary, "value": "", "value_raw": None, "unit": ""}
    return None


def _metric_value(metric: dict[str, Any]) -> str:
    value = _format_scalar(metric.get("value"))
    unit = _text(metric.get("unit"), "")
    return f"{value} {unit}".strip()


def _metric_bar(value: Any, unit: Any) -> str:
    percent = _metric_percent(value, unit)
    if percent is None:
        return ""
    width = f"{percent:.4g}"
    return f"""<div class="metric-bar" aria-hidden="true">
            <span class="metric-fill" style="width: {width}%;"></span>
          </div>"""


def _metric_percent(value: Any, unit: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    unit_text = _text(unit, "").strip().lower()
    if unit_text in {"ratio", "rate", "fraction"} and 0 <= numeric <= 1:
        percent = numeric * 100
    elif unit_text in {"%", "pct", "percent", "percentage"}:
        percent = numeric * 100 if 0 <= numeric <= 1 else numeric
    elif not unit_text and 0 <= numeric <= 1:
        percent = numeric * 100
    else:
        return None
    return min(100.0, max(0.0, percent))


def _definition(label: str, value: Any, default: str = "-") -> str:
    text = _text(value, default)
    if not text:
        return ""
    return f"<div><dt>{_escape(label)}</dt><dd>{_escape(text)}</dd></div>"


def _stat_card(label: str, value: Any, *, duration: bool = False) -> str:
    display = _format_duration(value) if duration else _format_scalar(value)
    return f"""<article class="stat-card">
      <span>{_escape(label)}</span>
      <strong>{_escape(display)}</strong>
    </article>"""


def _table(headers: list[str], rows: list[str], *, empty_message: str = "") -> str:
    if not rows:
        return f"<p class=\"muted\">{_escape(empty_message)}</p>" if empty_message else ""
    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    return f"""<div class="table-wrap"><table>
      <thead><tr>{head}</tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table></div>"""


def _kv_table(rows: list[tuple[str, Any]]) -> str:
    rendered = []
    for label, value in rows:
        if value in (None, "", [], {}):
            continue
        rendered.append(
            f"<tr><th>{_escape(label)}</th><td>{_escape(_format_compact(value))}</td></tr>"
        )
    if not rendered:
        return "<p class=\"muted\">No metadata.</p>"
    return f"<table class=\"kv-table\"><tbody>{''.join(rendered)}</tbody></table>"


def _badge(value: str, kind: str) -> str:
    normalized = _class_token(value)
    return f"<span class=\"badge {kind} {kind}-{normalized}\">{_escape(value)}</span>"


def _chips(values: Any) -> str:
    items = [_text(value, "") for value in _list(values) if _text(value, "")]
    return "".join(f"<span class=\"chip mono\">{_escape(item)}</span>" for item in items)


def _ids_line(label: str, values: Any, *, inline: bool = False) -> str:
    items = [_text(value, "") for value in _list(values) if _text(value, "")]
    if not items:
        return ""
    text = ", ".join(items[:5])
    if len(items) > 5:
        text += f" (+{len(items) - 5})"
    if inline:
        return f"{label}: {text}"
    return f"<p class=\"detail-line\"><strong>{_escape(label)}:</strong> {_escape(text)}</p>"


def _detail_text(value: str) -> str:
    return f"<p class=\"detail-line\">{_escape(value)}</p>" if value else ""


def _collapsible_debug(label: str, value: Any) -> str:
    return f"""<details class="preview">
      <summary>{_escape(label)}</summary>
      {_render_value(value)}
    </details>"""


def _render_value(value: Any) -> str:
    """Bounded fallback renderer for compact debug metadata."""
    if value in (None, {}, []):
        return ""
    clipped, was_truncated = _truncate(
        json.dumps(_sanitize_payload(value), ensure_ascii=False, indent=2, sort_keys=True)
        if isinstance(value, (dict, list))
        else str(value),
        _DEBUG_LIMIT,
    )
    suffix = "\n... truncated" if was_truncated else ""
    return f"<pre>{_escape(clipped + suffix)}</pre>"


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_payload(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(child) for child in value]
    if isinstance(value, str):
        return _redact_data_urls(value)
    return value


def _safe_script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return _format_scalar(value)
    return _format_compact(value)


def _format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.5g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _format_compact(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_text(item, "") for item in value[:6])
    if isinstance(value, dict):
        parts = []
        for key, child in sorted(value.items())[:6]:
            if isinstance(child, (dict, list)):
                parts.append(str(key))
            else:
                parts.append(f"{key}={_format_scalar(child)}")
        return ", ".join(parts)
    return _format_scalar(value)


def _format_duration(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return str(value)
    if seconds < 60:
        return f"{seconds:.3g}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(remainder)}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def _format_bytes(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        size = float(value)
    except (TypeError, ValueError):
        return str(value)
    units = ("B", "KB", "MB", "GB")
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1
    return f"{size:.4g} {units[index]}"


def _short_sha(value: Any) -> str:
    text = _text(value, "")
    return text[:12] if text else ""


def _truncate(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[:limit], True


def _class_token(value: str) -> str:
    token = "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")
    return token or "unknown"


def _redact_data_urls(value: str) -> str:
    value = _SVG_DATA_IMAGE_URL_RE.sub("<redacted:data-url>", value)
    return _DATA_IMAGE_URL_RE.sub("<redacted:data-url>", value)


def _safe_preview_source(value: str) -> str:
    text = _redact_data_urls(value)
    if text.startswith(("http://", "https://")):
        parts = urlsplit(text)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    return text


def _positive(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _escape(value: Any) -> str:
    return html.escape(_text(value, ""))


def _first_present(*args: Any) -> Any:
    mappings = [arg for arg in args if isinstance(arg, dict)]
    keys = [arg for arg in args if isinstance(arg, str)]
    for mapping in mappings:
        for key in keys:
            value = mapping.get(key)
            if value is not None and value != "":
                return value
    return None


_CSS = """
:root {
  color-scheme: light;
  --ink: #20242a;
  --muted: #65717f;
  --line: #d8dee6;
  --line-soft: #eceff3;
  --surface: #ffffff;
  --page: #f5f6f8;
  --accent: #1d4f91;
  --danger: #a33a33;
  --warn: #8a5b10;
  --ok: #276749;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--page);
  font-size: 14px;
  line-height: 1.45;
}
.report-shell {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
}
.hero {
  margin-top: 16px;
  padding: 18px 20px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.hero-topline,
.section-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.hero-topline {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
.hero-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.3fr) minmax(260px, .9fr) minmax(180px, .5fr);
  gap: 18px;
  align-items: start;
  margin-top: 10px;
}
h1, h2, h3, p { margin-top: 0; }
h1 {
  margin-bottom: 8px;
  font-size: 24px;
  line-height: 1.2;
  letter-spacing: 0;
}
h2 { margin-bottom: 0; font-size: 17px; letter-spacing: 0; }
h3 { margin-bottom: 8px; font-size: 14px; letter-spacing: 0; }
.muted { color: var(--muted); }
.section, .footer {
  margin-top: 14px;
  padding: 16px 20px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.subsection + .subsection { margin-top: 16px; }
.meta-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px 14px;
  margin: 0;
}
dt, .label, .stat-card span {
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
}
dd { margin: 2px 0 0; font-weight: 650; word-break: break-word; }
.primary-metric,
.metric-focus,
.stat-card,
.profile-panel,
.panel {
  border: 1px solid var(--line-soft);
  border-radius: 6px;
  background: #fbfcfd;
}
.primary-metric,
.metric-focus {
  padding: 12px;
}
.primary-metric strong,
.metric-focus strong {
  display: block;
  margin-top: 4px;
  font-size: 24px;
  line-height: 1.1;
}
.metric-focus { margin: 12px 0; max-width: 260px; }
.metric-bar {
  width: 100%;
  height: 6px;
  margin-top: 9px;
  overflow: hidden;
  background: #e5ebf1;
  border-radius: 999px;
}
.metric-fill {
  display: block;
  height: 100%;
  background: var(--accent);
  border-radius: inherit;
}
.stat-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(120px, 1fr));
  gap: 10px;
  margin-top: 12px;
}
.stat-card { padding: 10px 12px; }
.stat-card strong { display: block; margin-top: 3px; font-size: 18px; }
.badge,
.chip {
  display: inline-flex;
  align-items: center;
  max-width: 100%;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}
.chip {
  margin: 2px 4px 2px 0;
  color: #334155;
  background: #edf2f7;
  border: 1px solid #d9e2ec;
  line-height: 1.25;
  overflow-wrap: anywhere;
  white-space: normal;
  word-break: break-word;
}
.verdict-passed,
.verdict-completed,
.verdict-passed-with-warnings,
.severity-low {
  color: var(--ok);
  background: #e6f4ea;
  border: 1px solid #b7dec5;
}
.verdict-failed,
.severity-critical,
.severity-high {
  color: var(--danger);
  background: #fae9e7;
  border: 1px solid #efc4bf;
}
.verdict-aborted,
.verdict-degraded,
.severity-medium {
  color: var(--warn);
  background: #fff3d6;
  border: 1px solid #edd393;
}
.verdict-unknown,
.severity-info,
.severity-unknown {
  color: #475569;
  background: #eef2f6;
  border: 1px solid #d8dee6;
}
.table-wrap { overflow-x: auto; margin-top: 10px; }
table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
}
th, td {
  border-top: 1px solid var(--line-soft);
  padding: 8px;
  text-align: left;
  vertical-align: top;
  word-break: break-word;
}
thead th {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  background: #f8fafc;
}
.kv-table th { width: 190px; color: var(--muted); font-weight: 650; }
.numeric { text-align: right; font-variant-numeric: tabular-nums; }
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 12px;
}
.detail-line { margin: 0 0 4px; }
.profile-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 12px;
}
.profile-panel { padding: 12px; }
details { margin-top: 10px; }
summary {
  cursor: pointer;
  color: var(--ink);
  font-weight: 700;
}
.panel { padding: 10px 12px; }
.preview pre,
pre {
  max-height: 340px;
  overflow: auto;
  margin: 8px 0 0;
  padding: 10px;
  color: #1f2937;
  background: #f7f9fb;
  border: 1px solid var(--line-soft);
  border-radius: 6px;
  white-space: pre-wrap;
  word-break: break-word;
}
.footer { margin-bottom: 20px; }
@media (max-width: 900px) {
  .hero-grid,
  .profile-grid { grid-template-columns: 1fr; }
  .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .meta-list { grid-template-columns: 1fr; }
}
@media (max-width: 560px) {
  .report-shell { width: min(100% - 20px, 1180px); }
  .hero, .section, .footer { padding: 14px; }
  .stat-grid { grid-template-columns: 1fr; }
  h1 { font-size: 20px; }
}
"""

_JS = """
(function () {
  "use strict";
  document.documentElement.dataset.reportJs = "loaded";
})();
// report.js
"""
