from __future__ import annotations

import html
import json
from typing import Any

from gage_eval.reporting.rendering._context import normalize_context


class StaticReportRenderer:
    """Renders a self-contained static HTML report without network dependencies."""

    def render(self, context: dict[str, Any]) -> str:
        payload = normalize_context(context)
        context_json = _safe_script_json(payload)
        run = payload.get("run") or {}
        headline = payload.get("headline") or {}
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GAGE Run Report - {html.escape(str(run.get("run_id", "unknown")))}</title>
  <style>{_CSS}</style>
</head>
<body>
  <header>
    <p>GAGE Run Report</p>
    <h1>{html.escape(str(headline.get("one_line_summary", "Run report generated.")))}</h1>
    <dl>
      <div><dt>Run</dt><dd>{html.escape(str(run.get("run_id", "unknown")))}</dd></div>
      <div><dt>Verdict</dt><dd>{html.escape(str(headline.get("verdict", "unknown")))}</dd></div>
    </dl>
  </header>
  <main>
    {_section("Run Overview", [run, headline])}
    {_section("Metrics", payload.get("metrics"))}
    {_section("Attention Cases", payload.get("attention_cases"))}
    {_section("Case Details", payload.get("case_details"))}
    {_section("Outliers", payload.get("outliers"))}
    {_section("Failure Analysis", payload.get("failure_clusters") or payload.get("reason_code_counts"))}
    {_section("Evidence", payload.get("evidence_refs"))}
    {_section("Scenario Profiles", payload.get("scenario_profiles"))}
    {_section("Methodology", payload.get("methodology"))}
  </main>
  <script type="application/json" id="gage-report-context">{context_json}</script>
  <script>
  window.gage = window.gage || {{}};
  window.gage.report_context = JSON.parse(document.getElementById("gage-report-context").textContent);
  </script>
  <script data-asset-name="report.js">{_JS}</script>
</body>
</html>
"""


def _section(title: str, value: Any) -> str:
    return f"""<section data-filter-target="{html.escape(title.lower().replace(" ", "-"))}">
      <h2>{html.escape(title)}</h2>
      <details open>
        <summary>{html.escape(title)} details</summary>
        {_render_value(value)}
      </details>
    </section>"""


def _render_value(value: Any) -> str:
    if value in (None, {}, []):
        return "<p>No data available.</p>"
    if isinstance(value, list):
        items = "".join(f"<li>{_render_value(item)}</li>" for item in value)
        return f"<ul>{items}</ul>"
    if isinstance(value, dict):
        rows = "".join(
            f"<tr><th>{html.escape(str(key))}</th><td>{_render_value(child)}</td></tr>"
            for key, child in sorted(value.items())
        )
        return f"<table>{rows}</table>"
    return f"<span>{html.escape(str(value))}</span>"


def _safe_script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


_CSS = """
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
body { margin: 0; color: #1f2933; background: #f7f8fa; }
header, main { max-width: 1120px; margin: 0 auto; padding: 24px; }
header { background: #ffffff; border-bottom: 1px solid #d9dee7; }
header p { margin: 0 0 8px; color: #52606d; font-size: 13px; text-transform: uppercase; }
h1 { margin: 0 0 16px; font-size: 30px; line-height: 1.2; font-weight: 700; }
h2 { margin: 0 0 12px; font-size: 18px; }
dl { display: flex; gap: 24px; margin: 0; }
dt { font-size: 12px; color: #66788a; }
dd { margin: 2px 0 0; font-weight: 600; }
section { padding: 20px 0; border-bottom: 1px solid #d9dee7; }
details { background: #ffffff; border: 1px solid #d9dee7; border-radius: 6px; padding: 12px; }
summary { cursor: pointer; font-weight: 600; }
table { width: 100%; border-collapse: collapse; margin-top: 12px; }
th, td { border-top: 1px solid #edf0f4; padding: 8px; text-align: left; vertical-align: top; }
th { width: 220px; color: #52606d; font-weight: 600; }
ul { margin: 12px 0 0; padding-left: 20px; }
"""

_JS = """
(function () {
  "use strict";
  document.documentElement.dataset.reportJs = "loaded";
})();
// report.js
"""
