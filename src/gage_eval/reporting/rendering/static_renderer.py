from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from gage_eval.reporting.contracts.reason_codes import ReasonCodeRegistry
from gage_eval.reporting.rendering._context import normalize_context


_PREVIEW_LIMIT = 1800
_DEBUG_LIMIT = 1200
_ROW_LIMIT = 8
_EVIDENCE_INITIAL_LIMIT = 5
_EVIDENCE_GROUP_RENDER_LIMIT = 50
_METRIC_ROW_LIMIT = 100
_DIAGNOSTIC_PATH_LIMIT = 3
_PREFERRED_METRIC_VALUE_KEYS = (
    "score",
    "mean",
    "accuracy",
    "acc",
    "reward",
    "rate",
    "pass_rate",
    "resolve_rate",
)
_SVG_DATA_IMAGE_URL_RE = re.compile(
    r"data:image/svg\+xml(?:;[a-z0-9.+-]+(?:=[^\s\"'<>;,)\]}]+)?)*,"
    r"(?:.*?</svg>|[^\s\"')}\]]+)",
    re.IGNORECASE | re.DOTALL,
)
_DATA_IMAGE_URL_RE = re.compile(
    r"data:image/[a-z0-9.+-]+(?:;[a-z0-9.+-]+(?:=[^\s\"'<>;,)\]}]+)?)*,[^\s\"')}\]]+",
    re.IGNORECASE,
)
_LOCAL_RUN_PATH_RE = re.compile(
    r"/(?:Users|home|private|tmp|var)/[^\s\"'<>]*?/runs/",
)


class StaticReportRenderer:
    """Renders a self-contained static HTML report without network dependencies."""

    def render(self, context: dict[str, Any]) -> str:
        payload = _sanitize_payload(normalize_context(context))
        context_json = _safe_script_json(_embedded_context_payload(payload))
        run = _mapping(payload.get("run"))
        title = f"GAGE Run Report - {_text(run.get('run_id'), 'unknown')}"
        main_body = "\n".join(_main_sections(payload))
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
  {_render_nav(payload)}
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


def _main_sections(payload: dict[str, Any]) -> list[str]:
    quick_stats = _render_quick_stats(payload)
    findings = _render_key_findings(payload)
    metrics = _render_metrics_dashboard(payload)
    scenario_profile = _render_scenario_profile(payload)
    evidence = _render_evidence_explorer(payload)
    verdict = _text(_mapping(payload.get("headline")).get("verdict"), "").replace("-", "_")
    ordered = (
        [findings, quick_stats]
        if findings and verdict in {"failed", "aborted", "degraded"}
        else [quick_stats, findings]
    )
    ordered.extend([metrics, scenario_profile, evidence])
    return [section for section in ordered if section]


def _render_hero(payload: dict[str, Any]) -> str:
    run = _mapping(payload.get("run"))
    headline = _mapping(payload.get("headline"))
    verdict = _text(headline.get("verdict"), "unknown")
    summary = _text(headline.get("one_line_summary"), "Run report generated.")
    reason = _text(headline.get("verdict_reason"), "")
    metric = _primary_metric(headline, _list(payload.get("metrics")))
    scenario_context_html = _scenario_context_panel(payload)
    metric_html = ""
    if metric and not (_is_runtime_health_synthetic_metric(metric) and scenario_context_html):
        bar = _metric_bar(metric.get("value_raw"), metric.get("unit"))
        metric_html = f"""
        <div class="primary-metric">
          <span class="label">Primary metric</span>
          <strong>{_escape(metric["value"])}</strong>
          <span>{_escape(_metric_caption(metric))}</span>
          {bar}
        </div>"""
    else:
        metric_html = scenario_context_html or _task_context_panel(payload)
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
        {_definition("Run dir", _display_run_dir(run.get("run_dir")), "")}
        {_definition("Game", _game_kits_label(payload), "")}
        {_definition("Started", run.get("started_at") or run.get("started_at_iso"), "")}
        {_definition("Finished", run.get("finished_at") or run.get("finished_at_iso"), "")}
        {_definition("Duration", _format_duration(_run_duration(run, {})) or "-", "-")}
      </dl>
      {metric_html}
    </div>
  </header>"""


def _render_quick_stats(payload: dict[str, Any]) -> str:
    run = _mapping(payload.get("run"))
    headline = _mapping(payload.get("headline"))
    runtime = _mapping(payload.get("runtime_health"))
    observability = _mapping(payload.get("observability_health"))
    metrics = _list(payload.get("metrics"))
    verdict = _text(headline.get("verdict"), "")
    failed = runtime.get("failed_count")
    aborted = runtime.get("aborted_count")
    stats = [
        ("Samples", runtime.get("sample_count")),
        ("Completed", runtime.get("completed_count")),
        ("Duration", _run_duration(run, runtime)),
    ]
    if _positive(failed) or verdict not in {"passed", "completed"}:
        stats.append(("Failed", failed))
    if _positive(aborted) or verdict in {"aborted", "failed", "passed_with_warnings", "passed-with-warnings"}:
        stats.append(("Aborted", aborted))
    optional = [
        (
            "Cost",
            _first_present(runtime, observability, "total_cost", "cost", "agent_cost")
            or _metric_display_by_ids(metrics, {"total_cost", "cost_usd", "agent_cost"}),
        ),
        (
            "Tokens",
            _first_present(runtime, observability, "total_tokens", "tokens", "token_count")
            or _metric_display_by_ids(metrics, {"total_tokens", "tokens", "token_count"}),
        ),
        (
            "Latency",
            _first_present(runtime, observability, "latency_s", "duration_s", "elapsed_s")
            or _metric_display_by_ids(metrics, {"latency_s", "duration_s", "elapsed_s"}),
        ),
    ]
    cards = "".join(_stat_card(label, value, duration=label == "Duration") for label, value in stats)
    cards += "".join(
        _stat_card(label, value, duration=False)
        for label, value in optional
        if value is not None and value != ""
    )
    return f"""<section id="quick-stats" class="report-shell section" data-filter-target="quick-stats">
    <div class="section-heading">
      <h2>Quick stats</h2>
    </div>
    <div class="stat-grid">{cards}</div>
  </section>"""


def _render_key_findings(payload: dict[str, Any]) -> str:
    attention_cases = _list(payload.get("attention_cases"))
    failure_clusters = _list(payload.get("failure_clusters"))
    outliers = _list(payload.get("outliers"))
    task_failures = _task_failure_rows(_list(payload.get("tasks")))
    if not attention_cases and not failure_clusters and not outliers and not task_failures:
        return ""

    blocks = []
    if task_failures:
        blocks.append(_task_failures_table(task_failures))
    if attention_cases:
        blocks.append(_attention_cases_table(attention_cases, _mapping(payload.get("case_details"))))
    if failure_clusters:
        blocks.append(_failure_clusters_table(failure_clusters))
    if outliers:
        blocks.append(_outliers_table(outliers))
    return f"""<section id="key-findings" class="report-shell section" data-filter-target="key-findings">
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
        <span>{_escape(_metric_caption(primary))}</span>
        {bar}
      </div>"""

    rows = []
    for metric in metrics[:_METRIC_ROW_LIMIT]:
        item = _mapping(metric)
        rows.append(
            "<tr>"
            f"<td>{_escape(item.get('metric_id') or item.get('id') or '')}</td>"
            f"<td>{_escape(_metric_label(item))}</td>"
            f"<td>{_escape(item.get('scope') or '')}</td>"
            f"<td class=\"metric-value\">{_escape(_metric_table_value(item))}</td>"
            f"<td>{_escape(item.get('unit') or '')}</td>"
            f"{_owner_cell(item.get('task_id') or item.get('section_id') or '')}"
            "</tr>"
        )
    table = _table(
        ["Metric", "Name", "Scope", "Value", "Unit", "Owner"],
        rows,
        empty_message="No metrics recorded.",
    )
    if len(metrics) > _METRIC_ROW_LIMIT:
        omitted = len(metrics) - _METRIC_ROW_LIMIT
        table += (
            f'<p class="muted">{omitted} more metrics omitted from HTML; '
            'see <a href="report_context.json">report_context.json</a>.</p>'
        )
    return f"""<section id="metrics" class="report-shell section" data-filter-target="metrics">
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
    return f"""<section id="scenario-profile" class="report-shell section" data-filter-target="scenario-profile">
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
        kind = _text(item.get("kind"), "unknown")
        role = _text(item.get("artifact_role") or item.get("owner"), "")
        group_name = f"{kind} / {role}" if kind == "artifact" and role else kind
        groups[group_name].append(item)

    blocks = []
    for kind in sorted(groups):
        group_refs = groups[kind]
        visible = group_refs[:_EVIDENCE_INITIAL_LIMIT]
        rendered = group_refs[:_EVIDENCE_GROUP_RENDER_LIMIT]
        extra = rendered[_EVIDENCE_INITIAL_LIMIT:]
        omitted = max(0, len(group_refs) - len(rendered))
        table = _evidence_table(visible)
        if extra:
            remaining = len(group_refs) - _EVIDENCE_INITIAL_LIMIT
            table += f"""<details class="more-evidence">
          <summary>+ {remaining} more</summary>
          {_evidence_table(extra)}
          {_omitted_rows_note(omitted, "evidence refs") if omitted else ""}
        </details>"""
        elif omitted:
            table += _omitted_rows_note(omitted, "evidence refs")
        blocks.append(
            f"""<details class="panel" open>
        <summary>{_escape(kind)} evidence ({len(group_refs)})</summary>
        {table}
      </details>"""
        )
    return f"""<section id="evidence-explorer" class="report-shell section" data-filter-target="evidence-explorer">
    <div class="section-heading">
      <h2>Evidence explorer</h2>
    </div>
    {"".join(blocks)}
  </section>"""


def _omitted_rows_note(count: int, label: str) -> str:
    if count <= 0:
        return ""
    return (
        f'<p class="muted">{count} more {label} omitted from HTML; '
        'see <a href="report_context.json">report_context.json</a>.</p>'
    )


def _render_footer(payload: dict[str, Any]) -> str:
    methodology = _mapping(payload.get("methodology"))
    diagnostics = _mapping(payload.get("diagnostics"))
    schema = _mapping(payload.get("schema"))
    run = _mapping(payload.get("run"))
    generated_by = _mapping(schema.get("generated_by"))
    methodology_summary = _methodology_summary(methodology)
    diagnostics_summary = _diagnostics_summary(diagnostics)
    reason_glossary = _render_reason_glossary(payload)
    metadata_rows = [
        ("Schema", f"{schema.get('name', '')} v{schema.get('major', '')}.{schema.get('minor', '')}"),
        ("Renderer compat", schema.get("renderer_compat")),
        ("Generated by", generated_by.get("component")),
        ("Generator version", generated_by.get("version")),
        ("Run", run.get("run_id")),
        ("Report status", diagnostics.get("report_pack_status")),
    ]
    return f"""<footer id="methodology" class="report-shell footer" data-filter-target="methodology">
    <details>
      <summary>Methodology</summary>
      {methodology_summary}
    </details>
    <details open>
      <summary>Diagnostics</summary>
      {diagnostics_summary}
    </details>
    {reason_glossary}
    <details open>
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
            _ids_line("Preview artifact refs", details.get("artifact_preview_ref_ids"), evidence=True),
            _ids_line("Evidence refs", item.get("evidence_ref_ids") or details.get("evidence_ref_ids"), evidence=True),
            _ids_line("Sample/trial", [item.get("sample_id"), item.get("trial_id")]),
        ]
        if details.get("message_history_preview") or details.get("tool_call_summary"):
            detail_bits.append(_collapsible_debug("Details", details))
        rows.append(
            "<tr>"
            f"<td>{_escape(case_id)}</td>"
            f"<td>{_badge(_text(item.get('severity'), 'unknown'), 'severity')}</td>"
            f"<td>{_chips(item.get('reason_codes'), reason_codes=True)}</td>"
            f"<td>{_escape(_format_scalar(scoring.get('priority_score')))}</td>"
            f"<td>{''.join(bit for bit in detail_bits if bit)}</td>"
            "</tr>"
        )
    return f"""<div class="subsection">
      <h3>Attention cases</h3>
      {_table(
          ["Case", "Severity", "Reason codes", "Priority", "Details"],
          rows,
          table_class="findings-table attention-cases-table",
      )}
    </div>"""


def _task_failure_rows(tasks: list[Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task in tasks:
        item = _mapping(task)
        execution = _mapping(item.get("execution"))
        status = _text(item.get("status") or execution.get("status"), "").lower()
        failure = _mapping(execution.get("failure"))
        message = _text(
            failure.get("message") or execution.get("error_message") or item.get("error_message"),
            "",
        )
        failed_step = _text(execution.get("failed_step"), "")
        if status not in {"failed", "aborted", "error"} and not message and not failed_step:
            continue
        rows.append(
            {
                "task_id": _text(item.get("task_id") or item.get("id"), ""),
                "status": status or "failed",
                "failed_step": failed_step,
                "error_type": _text(failure.get("error_type") or execution.get("error_type"), ""),
                "message": message,
            }
        )
    return rows


def _task_failures_table(failures: list[dict[str, str]]) -> str:
    rows = []
    for item in failures[:_ROW_LIMIT]:
        details = [
            _detail_text(item.get("message", "")),
            _detail_text(f"Error type: {item['error_type']}" if item.get("error_type") else ""),
        ]
        rows.append(
            "<tr>"
            f"<td>{_escape(item.get('task_id') or '')}</td>"
            f"<td>{_badge(_text(item.get('status'), 'failed'), 'severity')}</td>"
            f"<td>{_escape(item.get('failed_step') or '')}</td>"
            f"<td>{''.join(bit for bit in details if bit)}</td>"
            "</tr>"
        )
    return f"""<div class="subsection">
      <h3>Task failures</h3>
      {_table(["Task", "Status", "Failed step", "Details"], rows, table_class="findings-table")}
    </div>"""


def _failure_clusters_table(clusters: list[Any]) -> str:
    rows = []
    for cluster in clusters[:_ROW_LIMIT]:
        item = _mapping(cluster)
        details = [
            _detail_text(_text(item.get("hypothesis"), "")),
            _detail_text(_text(item.get("recommended_action"), "")),
            _ids_line("Samples", item.get("sample_ids")),
            _ids_line("Evidence refs", item.get("representative_ref_ids"), evidence=True),
        ]
        rows.append(
            "<tr>"
            f"<td>{_escape(item.get('label') or item.get('cluster_id') or '')}</td>"
            f"<td>{_badge(_text(item.get('severity'), 'unknown'), 'severity')}</td>"
            f"<td>{_chips(item.get('cluster_key'), reason_codes=True)}</td>"
            f"<td class=\"numeric\">{_escape(_format_scalar(item.get('count')))}</td>"
            f"<td>{''.join(bit for bit in details if bit)}</td>"
            "</tr>"
        )
    return f"""<div class="subsection">
      <h3>Failure clusters</h3>
      {_table(["Cluster", "Severity", "Reason codes", "Count", "Details"], rows, table_class="findings-table")}
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
                        _escape(_text(value.get("sample_id"), "")),
                        _escape(_format_scalar(value.get("value"))),
                        _ids_line("refs", value.get("evidence_ref_ids"), inline=True, evidence=True),
                    )
                    if part
                )
            )
        rows.append(
            "<tr>"
            f"<td>{_escape(item.get('metric_id') or '')}</td>"
            f"<td>{_escape(item.get('scope') or '')}</td>"
            f"<td>{_escape(item.get('ranking') or '')}</td>"
            f"<td>{'; '.join(entries)}</td>"
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
    rows = {
        "agent": _agent_profile_rows,
        "external_harness": _external_harness_profile_rows,
        "game": _game_profile_rows,
    }[kind](profile)
    return f"""<article class="profile-panel">
      <h3>{titles[kind]}</h3>
      {_kv_table(rows)}
    </article>"""


def _agent_profile_rows(profile: dict[str, Any]) -> list[tuple[str, Any]]:
    return [
        ("Trial count", profile.get("trial_count")),
        ("Failed trial count", profile.get("failed_trial_count")),
        ("Representative refs", profile.get("representative_ref_ids")),
    ]


def _external_harness_profile_rows(profile: dict[str, Any]) -> list[tuple[str, Any]]:
    harnesses = _list(profile.get("harnesses"))
    sample_count = sum(_int_value(_mapping(item).get("sample_count")) for item in harnesses)
    harness_ids = [
        _text(_mapping(item).get("harness_id"), "")
        for item in harnesses
        if _text(_mapping(item).get("harness_id"), "")
    ]
    return [
        ("Harnesses", harness_ids),
        ("Sample count", sample_count if sample_count else None),
        ("Trial count", profile.get("trial_count")),
        ("Status rollup", profile.get("trial_rollup")),
        ("Representative refs", profile.get("representative_ref_ids")),
    ]


def _game_profile_rows(profile: dict[str, Any]) -> list[tuple[str, Any]]:
    return [
        ("Game kits", profile.get("game_kits")),
        ("Move count", profile.get("move_count")),
        ("Illegal actions", _illegal_actions_summary(profile.get("illegal_actions"), include_zero=True)),
        ("Replay refs", profile.get("replay_refs")),
    ]


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


def _evidence_table(refs: list[dict[str, Any]]) -> str:
    include_sample = any(ref.get("sample_id") for ref in refs)
    include_task = any(ref.get("task_id") for ref in refs)
    headers = ["Ref", "Path / preview", "Size", "SHA"]
    if include_sample:
        headers.append("Sample")
    if include_task:
        headers.append("Task")
    rows = []
    for ref in refs:
        preview = _preview_block(ref.get("preview"), ref.get("mime_type"))
        path = _text(ref.get("path"), "")
        media = _media_preview(ref)
        cells = [
            f"<td class=\"mono evidence-ref\">{_escape(ref.get('ref_id') or '')}</td>",
            f"<td class=\"evidence-path\">{_path_link(path)}{media}{preview}</td>",
            f"<td>{_escape(_format_bytes(ref.get('size_bytes')))}</td>",
            f"<td class=\"mono\" title=\"{_escape(ref.get('sha256') or '')}\">{_escape(_short_sha(ref.get('sha256')))}</td>",
        ]
        if include_sample:
            sample = _text(ref.get("sample_id"), "")
            cells.append(f"<td class=\"evidence-sample\" title=\"{_escape(sample)}\">{_escape(sample)}</td>")
        if include_task:
            task = _text(ref.get("task_id"), "")
            cells.append(f"<td class=\"evidence-task\" title=\"{_escape(task)}\">{_escape(task)}</td>")
        row_id = _evidence_anchor(ref.get("ref_id"))
        rows.append(f"<tr id=\"{_escape(row_id)}\">{''.join(cells)}</tr>")
    return _table(headers, rows)


def _media_preview(ref: dict[str, Any]) -> str:
    if _text(ref.get("kind"), "") != "media" or not _text(ref.get("mime_type"), "").startswith("image/"):
        return ""
    url = _safe_media_url(_mapping(ref.get("preview")))
    note = ""
    path = _text(ref.get("path"), "")
    if path.startswith("external://sha256/"):
        note = '<p class="muted media-note">External media URL digest; image source is not stored as base64 in the report.</p>'
    if not url:
        return note
    return (
        '<figure class="media-preview">'
        f'<img class="media-thumb" src="{_escape(url)}" alt="Media evidence preview" loading="lazy" referrerpolicy="no-referrer">'
        f"{note}</figure>"
    )


def _safe_media_url(preview: dict[str, Any]) -> str:
    for key in ("source", "url", "image_url"):
        value = preview.get(key)
        if not isinstance(value, str):
            continue
        sanitized = _safe_preview_source(value)
        parts = urlsplit(sanitized)
        if parts.scheme in {"http", "https"} and parts.netloc:
            return sanitized
        host_only = _host_only_media_url(sanitized)
        if host_only:
            return host_only
    return ""


def _host_only_media_url(value: str) -> str:
    if not value or value.startswith(("/", ".", "~")) or "://" in value:
        return ""
    head, _, tail = value.partition("/")
    if not tail or "." not in head or any(char.isspace() for char in head):
        return ""
    return f"https://{value}"


def _preview_block(preview: Any, mime_type: Any) -> str:
    if preview in (None, {}, []):
        return ""
    text = _preview_text(preview, _text(mime_type, ""))
    if not text:
        return ""
    clipped, was_truncated = _truncate(text, _PREVIEW_LIMIT)
    note = " <span class=\"muted\">truncated</span>" if was_truncated else ""
    preview_html = _preview_html(clipped, _text(mime_type, ""))
    return f"""<details class="preview">
        <summary>Preview{note}</summary>
        <pre>{preview_html}</pre>
      </details>"""


def _preview_text(preview: Any, mime_type: str) -> str:
    if isinstance(preview, str):
        return _safe_visible_text(preview)
    if isinstance(preview, dict):
        for key in ("text", "content", "json", "sample", "preview"):
            value = preview.get(key)
            if isinstance(value, str):
                return _safe_visible_text(value)
            if isinstance(value, (dict, list)):
                return _safe_visible_text(json.dumps(_sanitize_payload(value), ensure_ascii=False, indent=2, sort_keys=True))
        if mime_type.startswith("image/"):
            for key in ("source", "url", "image_url"):
                value = preview.get(key)
                if isinstance(value, str):
                    return _safe_preview_source(value)
        if "json" in mime_type or "text" in mime_type:
            return _safe_visible_text(json.dumps(_sanitize_payload(preview), ensure_ascii=False, indent=2, sort_keys=True))
    if isinstance(preview, list) and ("json" in mime_type or "text" in mime_type):
        return _safe_visible_text(json.dumps(_sanitize_payload(preview), ensure_ascii=False, indent=2, sort_keys=True))
    return ""


def _preview_html(text: str, mime_type: str) -> str:
    escaped = _escape(text)
    if _looks_json_preview(text, mime_type):
        return _highlight_json_escaped(escaped)
    return escaped


def _looks_json_preview(text: str, mime_type: str) -> bool:
    stripped = text.lstrip()
    return "json" in mime_type.lower() or stripped.startswith("{") or stripped.startswith("[")


def _highlight_json_escaped(escaped: str) -> str:
    highlighted = re.sub(
        r"(&quot;[^&<>]*?&quot;)(\s*:)",
        r'<span class="json-key">\1</span>\2',
        escaped,
    )
    highlighted = re.sub(
        r"(:\s*)(&quot;[^&<>]*?&quot;)",
        r'\1<span class="json-string">\2</span>',
        highlighted,
    )
    highlighted = re.sub(
        r"(:\s*)(-?\d+(?:\.\d+)?)",
        r'\1<span class="json-number">\2</span>',
        highlighted,
    )
    highlighted = re.sub(
        r"(:\s*)(true|false|null)\b",
        r'\1<span class="json-bool">\2</span>',
        highlighted,
    )
    return highlighted


def _methodology_summary(methodology: dict[str, Any]) -> str:
    if not methodology:
        return "<p class=\"muted\">Methodology metadata was not provided.</p>"
    rows = []
    for key, value in sorted(methodology.items()):
        if key == "run_metadata" and isinstance(value, dict):
            rows.append(("Run Metadata", f"{len(value)} fields; see report_context.json"))
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            rows.append((key.replace("_", " ").title(), value))
        elif isinstance(value, list):
            list_value = _unique_list(value) if key == "metric_ids" else value
            rows.append((key.replace("_", " ").title(), ", ".join(_text(item, "") for item in list_value[:6])))
        elif isinstance(value, dict):
            rows.append((key.replace("_", " ").title(), ", ".join(sorted(map(str, value.keys()))[:6])))
    return _kv_table(rows)


def _unique_list(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = _text(value, "")
        if key in seen:
            continue
        result.append(value)
        seen.add(key)
    return result


def _diagnostics_summary(diagnostics: dict[str, Any]) -> str:
    if not diagnostics:
        return "<p class=\"muted\">No diagnostics metadata was provided.</p>"
    warnings_raw = _list(diagnostics.get("warnings"))
    routine_redactions, visible_warnings = _partition_routine_redaction_warnings(warnings_raw)
    warnings = _diagnostic_rows(visible_warnings, severity="warning")
    errors = _diagnostic_rows(_list(diagnostics.get("errors")), severity="error")
    rows = [
        ("Report Pack Status", diagnostics.get("report_pack_status")),
        ("Warnings", len(warnings)),
        ("Privacy Redactions", _routine_redaction_summary(routine_redactions) if routine_redactions else None),
        ("Errors", len(errors)),
    ]
    html = [_kv_table(rows)]
    if warnings:
        html.append("<h3>Warnings</h3>" + _diagnostic_table(warnings))
    if errors:
        html.append("<h3>Errors</h3>" + _diagnostic_table(errors))
    return "".join(html)


_ROUTINE_REDACTION_PATHS = {
    "report_context.md",
    "report.html",
    "prompt.txt",
}


def _partition_routine_redaction_warnings(items: list[Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    routine: list[dict[str, Any]] = []
    visible: list[Any] = []
    for item in items:
        value = _mapping(item)
        if (
            value.get("code") == "report_pack.secret_redacted"
            and _text(value.get("path"), "") in _ROUTINE_REDACTION_PATHS
        ):
            routine.append(value)
            continue
        visible.append(item)
    return routine, visible


def _routine_redaction_summary(items: list[dict[str, Any]]) -> str:
    asset_count = len(items)
    finding_count = sum(_int(item.get("finding_count")) for item in items)
    asset_label = "asset" if asset_count == 1 else "assets"
    if finding_count:
        return f"{asset_count} rendered {asset_label}; {finding_count} findings redacted"
    return f"{asset_count} rendered {asset_label}"


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _diagnostic_rows(items: list[Any], *, severity: str) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        value = _mapping(item)
        code = _text(value.get("code"), "")
        message = _text(value.get("message") or _diagnostic_explanation(code), "")
        key = (code, message)
        row = grouped.setdefault(
            key,
            {
                "severity": severity,
                "code": code,
                "paths": [],
                "finding_count": 0,
                "occurrences": 0,
                "message": message,
            },
        )
        row["occurrences"] += 1
        path = _text(value.get("path"), "")
        if path:
            row["paths"].append(path)
        row["finding_count"] += _int(value.get("finding_count"))
    rows: list[dict[str, str]] = []
    for row in grouped.values():
        rows.append(
            {
                "severity": row["severity"],
                "code": row["code"],
                "path_html": _diagnostic_paths_summary(row["paths"]),
                "finding_count": _text(row["finding_count"], "") if row["finding_count"] else "",
                "occurrences": _text(row["occurrences"], ""),
                "message": row["message"],
            }
        )
    return rows


def _diagnostic_table(items: list[dict[str, str]]) -> str:
    rows = []
    for item in items:
        rows.append(
            "<tr>"
            f"<td>{_badge(item['severity'], 'severity')}</td>"
            f"<td class=\"mono\">{_escape(item.get('code') or '')}</td>"
            f"<td class=\"numeric\">{_escape(item.get('occurrences') or '')}</td>"
            f"<td>{item.get('path_html') or ''}</td>"
            f"<td class=\"numeric\">{_escape(item.get('finding_count') or '')}</td>"
            f"<td>{_escape(item.get('message') or '')}</td>"
            "</tr>"
        )
    return _table(["Level", "Code", "Occurrences", "Paths", "Findings", "Meaning"], rows, table_class="diagnostics-table")


def _diagnostic_explanation(code: str) -> str:
    explanations = {
        "report_pack.secret_redacted": "Report-visible output contained sensitive-looking content and was redacted before writing.",
        "report_pack.file_missing": "An optional source file was missing; the report was generated with available evidence.",
        "report_pack.derived_sample_detail_without_journal_record": "Derived sample detail files were present without matching journal records; the report uses available sample evidence.",
    }
    return explanations.get(code, "")


def _diagnostic_paths_summary(paths: list[str]) -> str:
    unique = _unique_list(paths)
    if not unique:
        return ""
    links = [_diagnostic_path_link(path) for path in unique[:_DIAGNOSTIC_PATH_LIMIT]]
    if len(unique) > _DIAGNOSTIC_PATH_LIMIT:
        links.append(_escape(f"+{len(unique) - _DIAGNOSTIC_PATH_LIMIT} more"))
    return ", ".join(links)


def _diagnostic_path_link(path: str) -> str:
    if not path:
        return ""
    if path in {"report.html", "report_context.md", "report_context.json", "diagnostics.json", "assets_manifest.json", "prompt.txt"}:
        return f'<a href="{_escape(quote(path, safe="/._-~:@"))}">{_escape(path)}</a>'
    return _path_link(path)


def _render_reason_glossary(payload: dict[str, Any]) -> str:
    codes = _collect_reason_codes(payload)
    if not codes:
        return ""
    try:
        registry = ReasonCodeRegistry.load_builtin()
    except Exception:
        registry = None

    rows = []
    for code in codes:
        try:
            entry = registry.get(code) if registry is not None else None
        except KeyError:
            entry = None
        if entry is None:
            english = "Unregistered reason code"
            chinese = ""
            impact = ""
            actionability = ""
        else:
            english = entry.human_readable_en
            chinese = entry.human_readable_zh
            impact = entry.impact_default
            actionability = entry.actionability_default
        rows.append(
            f"<tr id=\"reason-{_class_token(code)}\">"
            f"<td class=\"mono\">{_escape(code)}</td>"
            f"<td>{_escape(english)}</td>"
            f"<td>{_escape(chinese)}</td>"
            f"<td>{_escape(impact)}</td>"
            f"<td>{_escape(actionability)}</td>"
            "</tr>"
        )
    return f"""<details id="reason-codes-glossary" open>
      <summary>Reason codes glossary</summary>
      {_table(["Code", "English", "中文", "Impact", "Actionability"], rows)}
    </details>"""


def _collect_reason_codes(payload: dict[str, Any]) -> list[str]:
    codes: set[str] = set()
    for case in _list(payload.get("attention_cases")):
        codes.update(_text(code, "") for code in _list(_mapping(case).get("reason_codes")))
    for cluster in _list(payload.get("failure_clusters")):
        codes.update(_text(code, "") for code in _list(_mapping(cluster).get("cluster_key")))
    return sorted(code for code in codes if code)


def _primary_metric(headline: dict[str, Any], metrics: list[Any]) -> dict[str, Any] | None:
    primary = headline.get("primary_metric")
    metric = _mapping(primary)
    if metric:
        value = _metric_value(metric)
        if not value:
            return None
        return {
            "metric_id": metric.get("metric_id") or metric.get("id"),
            "label": _metric_label(metric),
            "value": value,
            "value_raw": _metric_raw_value(metric),
            "unit": metric.get("unit"),
            "source": metric.get("source"),
            "synthetic": metric.get("synthetic"),
        }
    if isinstance(primary, str):
        for item in metrics:
            metric = _mapping(item)
            if primary in {metric.get("metric_id"), metric.get("id"), metric.get("name")}:
                value = _metric_value(metric)
                if not value:
                    return None
                return {
                    "metric_id": metric.get("metric_id") or metric.get("id"),
                    "label": _metric_label(metric),
                    "value": value,
                    "value_raw": _metric_raw_value(metric),
                    "unit": metric.get("unit"),
                    "source": metric.get("source"),
                    "synthetic": metric.get("synthetic"),
                }
        return None
    return None


def _is_runtime_health_synthetic_metric(metric: dict[str, Any] | None) -> bool:
    if not metric:
        return False
    metric_id = _text(metric.get("metric_id"), "")
    return (
        metric.get("synthetic") is True
        or metric.get("source") == "runtime_health"
        or metric_id in {"sample_completion_rate", "task_success_rate"}
    )


def _metric_value(metric: dict[str, Any]) -> str:
    return _metric_table_value(metric)


def _metric_caption(metric: dict[str, Any]) -> str:
    label = _text(metric.get("label"), "")
    unit = _text(metric.get("unit"), "").strip()
    if unit and label:
        return f"{unit} · {label}"
    return label or unit


def _metric_label(metric: dict[str, Any]) -> str:
    label = _text(metric.get("name") or metric.get("label"), "")
    if label:
        return label
    metric_id = _text(metric.get("metric_id") or metric.get("id"), "")
    return _titleize_identifier(metric_id) if metric_id else ""


def _owner_cell(value: Any) -> str:
    text = _text(value, "")
    return f"<td class=\"metric-owner\" title=\"{_escape(text)}\">{_escape(text)}</td>"


def _path_link(path: str) -> str:
    if not path:
        return ""
    href = _local_artifact_href(path)
    text = _escape(path)
    if not href:
        return text
    return f"<a href=\"{_escape(href)}\" title=\"{text}\">{text}</a>"


def _local_artifact_href(path: str) -> str:
    if not path or path.startswith("/") or path.startswith("#"):
        return ""
    parts = urlsplit(path)
    if parts.scheme or parts.netloc:
        return ""
    if not path.startswith(
        (
            "artifacts/",
            "replays/",
            "external_harness/",
            "samples/",
            "samples.jsonl",
            "events.jsonl",
            "summary.json",
        )
    ):
        return ""
    return "../" + quote(path, safe="/._-~:@")


def _metric_table_value(metric: dict[str, Any]) -> str:
    if "value" in metric:
        return _format_metric_scalar(metric.get("value"))
    values = _mapping(metric.get("values"))
    if not values:
        return ""
    raw_values = _mapping(metric.get("raw_values"))
    for key in _PREFERRED_METRIC_VALUE_KEYS:
        if key in values:
            raw_value = raw_values.get(key)
            return _format_metric_scalar(raw_value) if isinstance(raw_value, (int, float)) else _format_metric_scalar(values[key])
    key, value = next(iter(values.items()))
    return f"{key}={_format_metric_scalar(value)}"


def _metric_display_by_ids(metrics: list[Any], metric_ids: set[str]) -> str | None:
    for metric in metrics:
        item = _mapping(metric)
        metric_id = _text(item.get("metric_id") or item.get("id") or item.get("name"), "")
        if metric_id in metric_ids:
            value = _metric_table_value(item)
            return value if value else None
    return None


def _metric_raw_value(metric: dict[str, Any]) -> Any:
    if "value" in metric:
        return metric.get("value")
    raw_values = _mapping(metric.get("raw_values"))
    for key in _PREFERRED_METRIC_VALUE_KEYS:
        if key in raw_values:
            return raw_values[key]
    if raw_values:
        return next(iter(raw_values.values()))
    values = _mapping(metric.get("values"))
    for key in _PREFERRED_METRIC_VALUE_KEYS:
        if key in values:
            return values[key]
    if values:
        return next(iter(values.values()))
    return None


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
    if unit_text in {"ratio", "rate", "fraction", "score"} and 0 <= numeric <= 1:
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


def _hero_context_panel(payload: dict[str, Any]) -> str:
    return _scenario_context_panel(payload) or _task_context_panel(payload)


def _scenario_context_panel(payload: dict[str, Any]) -> str:
    profiles = _mapping(payload.get("scenario_profiles"))
    external = _mapping(profiles.get("external_harness"))
    harnesses = _list(external.get("harnesses"))
    if harnesses:
        rows = _external_harness_profile_rows(external)[:3]
        return f"""<div class="primary-metric context-panel">
          <span class="label">Run context</span>
          {_kv_table(rows)}
        </div>"""
    game = _mapping(profiles.get("game"))
    if _profile_has_signal("game", game):
        illegal_actions = _illegal_actions_summary(game.get("illegal_actions"), include_zero=False)
        rows = [
            ("Move count", game.get("move_count")),
            ("Illegal actions", illegal_actions),
        ]
        return f"""<div class="primary-metric context-panel">
          <span class="label">Run context</span>
          {_kv_table(rows)}
        </div>"""
    return ""


def _task_context_panel(payload: dict[str, Any]) -> str:
    tasks = _list(payload.get("tasks"))
    if tasks:
        first = _mapping(tasks[0])
        rows = [
            ("Task", first.get("task_id") or first.get("id")),
            ("Status", first.get("status")),
        ]
        return f"""<div class="primary-metric context-panel">
          <span class="label">Run context</span>
          {_kv_table(rows)}
        </div>"""
    return ""


def _game_kits_label(payload: dict[str, Any]) -> str:
    game = _mapping(_mapping(payload.get("scenario_profiles")).get("game"))
    kits = [_text(item, "") for item in _list(game.get("game_kits")) if _text(item, "")]
    return ", ".join(kits)


def _illegal_actions_summary(value: Any, *, include_zero: bool) -> str:
    illegal = _mapping(value)
    total = _int_value(illegal.get("total"))
    games = _int_value(illegal.get("games"))
    if total <= 0 and games <= 0:
        return "0" if include_zero else ""
    if games > 0:
        return f"{total} total / {games} games"
    return f"{total} total"


def _render_nav(payload: dict[str, Any]) -> str:
    links = [("Quick stats", "quick-stats"), ("Metrics", "metrics")]
    if payload.get("attention_cases") or payload.get("failure_clusters") or payload.get("outliers"):
        links.append(("Findings", "key-findings"))
    if _has_scenario_profile_signal(payload):
        links.append(("Profiles", "scenario-profile"))
    if payload.get("evidence_refs"):
        links.append(("Evidence", "evidence-explorer"))
    links.append(("Diagnostics", "methodology"))
    return "<nav class=\"report-shell report-nav\" aria-label=\"Report sections\">" + "".join(
        f"<a href=\"#{_escape(target)}\">{_escape(label)}</a>" for label, target in links
    ) + "</nav>"


def _has_scenario_profile_signal(payload: dict[str, Any]) -> bool:
    profiles = _mapping(payload.get("scenario_profiles"))
    return any(
        _profile_has_signal(str(kind), _mapping(profile))
        for kind, profile in profiles.items()
    )


def _stat_card(label: str, value: Any, *, duration: bool = False) -> str:
    display = _format_duration(value) if duration else _format_scalar(value)
    if not display:
        display = "-"
    return f"""<article class="stat-card">
      <span>{_escape(label)}</span>
      <strong>{_escape(display)}</strong>
    </article>"""


def _table(headers: list[str], rows: list[str], *, empty_message: str = "", table_class: str = "") -> str:
    if not rows:
        return f"<p class=\"muted\">{_escape(empty_message)}</p>" if empty_message else ""
    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    class_attr = f' class="{_escape(table_class)}"' if table_class else ""
    return f"""<div class="table-wrap"><table{class_attr}>
      <thead><tr>{head}</tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table></div>"""


def _kv_table(rows: list[tuple[str, Any]]) -> str:
    rendered = []
    for label, value in rows:
        if value in (None, "", [], {}):
            continue
        rendered.append(
            f"<tr><th>{_escape(label)}</th><td>{_kv_value(label, value)}</td></tr>"
        )
    if not rendered:
        return "<p class=\"muted\">No metadata.</p>"
    return f"<table class=\"kv-table\"><tbody>{''.join(rendered)}</tbody></table>"


def _kv_value(label: str, value: Any) -> str:
    if _is_evidence_refs_label(label):
        refs = [_text(item, "") for item in _list(value) if _text(item, "")]
        if refs:
            return ", ".join(_evidence_link(ref) for ref in refs[:5])
    return _escape(_format_compact(value))


def _is_evidence_refs_label(label: str) -> bool:
    normalized = label.strip().lower()
    return normalized.endswith("refs") or normalized.endswith("ref ids")


def _badge(value: str, kind: str) -> str:
    normalized = _class_token(value)
    label = _badge_label(value, kind)
    return f"<span class=\"badge {kind} {kind}-{normalized}\">{_escape(label)}</span>"


def _badge_label(value: str, kind: str) -> str:
    if kind == "verdict":
        labels = {
            "passed": "completed",
            "passed_with_warnings": "completed with warnings",
            "passed-with-warnings": "completed with warnings",
        }
        return labels.get(value, value)
    return value


def _chips(values: Any, *, reason_codes: bool = False) -> str:
    items = [_text(value, "") for value in _list(values) if _text(value, "")]
    if reason_codes:
        return "".join(
            f"<a class=\"chip mono\" href=\"#reason-{_class_token(item)}\">{_escape(item)}</a>"
            for item in items
        )
    return "".join(f"<span class=\"chip mono\">{_escape(item)}</span>" for item in items)


def _ids_line(label: str, values: Any, *, inline: bool = False, evidence: bool = False) -> str:
    items = [_text(value, "") for value in _list(values) if _text(value, "")]
    if not items:
        return ""
    if evidence:
        text = ", ".join(_evidence_link(item) for item in items[:5])
    else:
        text = _escape(", ".join(items[:5]))
    if len(items) > 5:
        text += _escape(f" (+{len(items) - 5})")
    if inline:
        return f"{label}: {text}"
    return f"<p class=\"detail-line\"><strong>{_escape(label)}:</strong> {text}</p>"


def _evidence_link(ref_id: str) -> str:
    text = _escape(ref_id)
    return f"<a class=\"evidence-ref-link\" href=\"#{_escape(_evidence_anchor(ref_id))}\">{text}</a>"


def _evidence_anchor(ref_id: Any) -> str:
    return "evidence-" + _class_token(_text(ref_id, "unknown"))


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
        return _safe_visible_text(value)
    return value


def _embedded_context_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the inline JSON useful for scripts without duplicating full evidence payloads."""
    return {
        "schema": payload.get("schema"),
        "run": {
            "run_id": _mapping(payload.get("run")).get("run_id"),
            "run_dir": _display_run_dir(_mapping(payload.get("run")).get("run_dir")),
        },
        "headline": payload.get("headline"),
        "diagnostics": {
            "report_pack_status": _mapping(payload.get("diagnostics")).get("report_pack_status"),
            "profile_ref_resolution_miss_count": _mapping(payload.get("diagnostics")).get(
                "profile_ref_resolution_miss_count"
            ),
            "warning_count": len(_list(_mapping(payload.get("diagnostics")).get("warnings"))),
            "error_count": len(_list(_mapping(payload.get("diagnostics")).get("errors"))),
        },
        "counts": {
            "metrics": len(_list(payload.get("metrics"))),
            "evidence_refs": len(_list(payload.get("evidence_refs"))),
            "attention_cases": len(_list(payload.get("attention_cases"))),
            "failure_clusters": len(_list(payload.get("failure_clusters"))),
            "outliers": len(_list(payload.get("outliers"))),
        },
        "full_context_ref": "report_context.json",
        "embedded_context_truncated": True,
    }


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


def _format_metric_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.5f}"
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


def _run_duration(run: dict[str, Any], runtime: dict[str, Any]) -> Any:
    timings = _mapping(run.get("timings"))
    return (
        runtime.get("duration_s")
        or run.get("duration_s")
        or timings.get("wall_runtime_s")
        or timings.get("runtime_s")
        or timings.get("total_s")
    )


def _titleize_identifier(value: str) -> str:
    if not value:
        return ""
    return " ".join(part.capitalize() for part in value.replace("-", "_").split("_") if part)


def _display_run_dir(value: Any) -> str:
    text = _text(value, "")
    if not text:
        return ""
    marker = "/runs/"
    normalized = text.replace("\\", "/")
    if marker in normalized:
        return "runs/" + normalized.split(marker, 1)[1]
    if normalized.startswith("runs/"):
        return normalized
    return normalized


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
    token = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return token or "unknown"


def _redact_data_urls(value: str) -> str:
    value = _SVG_DATA_IMAGE_URL_RE.sub("<redacted:data-url>", value)
    return _DATA_IMAGE_URL_RE.sub("<redacted:data-url>", value)


def _safe_visible_text(value: str) -> str:
    return _relativize_run_paths(_redact_data_urls(value))


def _relativize_run_paths(value: str) -> str:
    value = _LOCAL_RUN_PATH_RE.sub("runs/", value)
    cwd = Path.cwd().resolve().as_posix()
    if cwd and cwd in value:
        value = value.replace(f"{cwd}/", f"{Path(cwd).name}/")
        value = value.replace(cwd, Path(cwd).name)
    home = Path.home().resolve().as_posix()
    if home and home in value:
        value = value.replace(f"{home}/", "<redacted:home>/")
        value = value.replace(home, "<redacted:home>")
    return value


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


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
  color-scheme: light dark;
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
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
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
  border-top: 3px solid var(--accent);
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
  font-size: 32px;
  line-height: 1.1;
  overflow-wrap: anywhere;
  word-break: break-word;
}
.metric-focus { margin: 12px 0; max-width: 260px; }
.context-panel .kv-table th { width: 110px; }
.report-nav {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
  padding: 8px 0;
  background: var(--page);
  border-bottom: 1px solid var(--line-soft);
}
.report-nav a {
  color: var(--accent);
  text-decoration: none;
  border: 1px solid #c8d8ec;
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 12px;
  font-weight: 700;
  background: #f4f8fd;
}
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
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
  margin-top: 12px;
}
.stat-card { padding: 10px 12px; }
.stat-card strong { display: block; margin-top: 3px; font-size: 26px; }
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
  text-decoration: none;
  line-height: 1.25;
  overflow-wrap: anywhere;
  white-space: normal;
  word-break: break-word;
}
.verdict-passed,
.verdict-completed,
.verdict-passed-with-warnings,
.severity-low {
  color: #14532d;
  background: #d9f2df;
  border: 1px solid #7fba8e;
}
.verdict-failed,
.severity-critical,
.severity-high,
.severity-error {
  color: var(--danger);
  background: #fae9e7;
  border: 1px solid #efc4bf;
}
.verdict-aborted,
.verdict-degraded,
.severity-medium,
.severity-warning {
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
  table-layout: auto;
}
th, td {
  border-top: 1px solid var(--line-soft);
  padding: 8px;
  text-align: left;
  vertical-align: top;
  overflow-wrap: anywhere;
  word-break: break-word;
}
thead th {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  background: #f8fafc;
}
.kv-table th { width: 190px; color: var(--muted); font-weight: 650; }
.findings-table th { white-space: nowrap; }
.attention-cases-table th:nth-child(4),
.attention-cases-table td:nth-child(4) {
  width: 96px;
  min-width: 96px;
  max-width: 96px;
  text-align: right;
  white-space: nowrap;
}
.attention-cases-table td:nth-child(4) {
  font-variant-numeric: tabular-nums;
}
.numeric { text-align: right; font-variant-numeric: tabular-nums; }
.metric-value { text-align: left; font-variant-numeric: tabular-nums; }
.metric-owner { max-width: 220px; overflow-wrap: anywhere; }
.evidence-path,
.evidence-sample,
.evidence-task {
  max-width: 200px;
  overflow-wrap: anywhere;
  word-break: break-all;
}
.evidence-path a {
  color: var(--accent);
  text-decoration: none;
}
.evidence-path a:hover {
  text-decoration: underline;
}
.evidence-ref-link {
  color: var(--accent);
  text-decoration: none;
}
.evidence-ref-link:hover { text-decoration: underline; }
.media-preview {
  margin: 8px 0;
}
.media-thumb {
  display: block;
  max-width: min(100%, 420px);
  max-height: 280px;
  border: 1px solid var(--line-soft);
  border-radius: 6px;
  object-fit: contain;
  background: #fff;
}
.media-note { margin: 6px 0 0; font-size: 12px; }
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 12px;
  overflow-wrap: anywhere;
}
.detail-line { margin: 0 0 4px; }
.profile-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
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
  max-height: 240px;
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
.json-key { color: #1d4f91; font-weight: 700; }
.json-string { color: #8a4f18; }
.json-number { color: #6d3bbd; }
.json-bool { color: #276749; font-weight: 700; }
.more-evidence { margin-top: 8px; }
.evidence-ref { min-width: 160px; word-break: break-all; }
.footer { margin-bottom: 20px; }
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #e8edf3;
    --muted: #9aa8b6;
    --line: #384453;
    --line-soft: #2c3642;
    --surface: #111820;
    --page: #0b1117;
    --accent: #7db4ff;
    --danger: #ff9c92;
    --warn: #f4c36a;
    --ok: #8bd49c;
  }
  .primary-metric,
  .metric-focus,
  .stat-card,
  .profile-panel,
  .panel {
    background: #141d26;
  }
  .report-nav a {
    color: #d6e8ff;
    background: #17283b;
    border-color: #315276;
  }
  thead th,
  .preview pre,
  pre {
    color: var(--ink);
    background: #0f1620;
  }
  .chip {
    color: #dbeafe;
    background: #17283b;
    border-color: #315276;
  }
  .verdict-passed,
  .verdict-completed,
  .verdict-passed-with-warnings,
  .severity-low {
    color: #d7ffe0;
    background: #12371f;
    border-color: #2e7742;
  }
  .verdict-failed,
  .severity-critical,
  .severity-high {
    color: #ffe2de;
    background: #3b1515;
    border-color: #7f3430;
  }
  .verdict-aborted,
  .verdict-degraded,
  .severity-medium {
    color: #fff1c2;
    background: #3b2a0f;
    border-color: #80601e;
  }
  .json-key { color: #9dccff; }
  .json-string { color: #f4b782; }
  .json-number { color: #d9b8ff; }
  .json-bool { color: #8bd49c; }
}
@media print {
  body { background: #fff; }
  .report-nav, script { display: none !important; }
  .report-shell { width: 100%; }
  .hero, .section, .footer { break-inside: avoid; box-shadow: none; }
  details { display: block; }
  details > summary { display: block; }
  .preview pre { max-height: none; overflow: visible; }
}
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
