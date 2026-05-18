from __future__ import annotations

import pytest

from gage_eval.reporting.contracts import ReportContext
from gage_eval.reporting.rendering.static_renderer import StaticReportRenderer


@pytest.mark.fast
def test_static_renderer_outputs_semantic_minimal_report_without_cdn() -> None:
    context = ReportContext.minimal("run").to_dict()
    html = StaticReportRenderer().render(context)

    for text in [
        "Run report",
        "Quick stats",
        "Metrics",
        "Methodology",
    ]:
        assert text in html
    for empty_section in [
        "Key findings",
        "Evidence explorer",
        "Scenario profile",
        "No data available",
    ]:
        assert empty_section not in html
    assert 'src="https://' not in html
    assert 'href="https://' not in html
    assert "cdn" not in html.lower()
    assert "<main>" in html
    assert "</main>" in html
    assert 'data-filter-target="metrics"' in html
    assert 'class="badge verdict verdict-passed"' in html
    assert "gage.report_context" in html


@pytest.mark.fast
def test_static_renderer_escapes_html() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["headline"]["one_line_summary"] = "<script>alert(1)</script>"
    context["attention_cases"] = [
        {
            "case_id": "case-1",
            "severity": "high",
            "reason_codes": ["scheduler.failed"],
            "summary": "<b>bad</b>",
            "evidence_ref_ids": ["evidence://artifact/log"],
            "scoring": {
                "frequency": 1.0,
                "impact": "high",
                "actionability": "high",
                "priority_score": 1.0,
            },
        }
    ]

    html = StaticReportRenderer().render(context)

    assert "<script>alert(1)</script>" not in html
    assert "<b>bad</b>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;b&gt;bad&lt;/b&gt;" in html


@pytest.mark.fast
def test_static_renderer_outputs_findings_evidence_previews_and_badges() -> None:
    context = _rich_context()

    html = StaticReportRenderer().render(context)

    assert "Key findings" in html
    assert 'class="badge severity severity-critical"' in html
    assert 'class="badge severity severity-high"' in html
    assert "scheduler.failed" in html
    assert "Evidence refs" in html
    assert "Evidence explorer" in html
    assert "<pre" in html
    assert "hello &lt;world&gt;" in html
    assert "AAAAAAAAAA" in html
    assert "truncated" in html
    assert 'data-filter-target="key-findings"' in html
    assert 'data-filter-target="evidence-explorer"' in html


@pytest.mark.fast
def test_static_renderer_summarizes_scenario_profiles_by_kind() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["scenario_profiles"] = {
        "agent": {
            "sample_count": 3,
            "failed_trial_count": 1,
            "representative_ref_ids": ["evidence://artifact/abc"],
            "tool_call_count": 12,
        },
        "external_harness": {
            "harness_name": "harbor",
            "failed_count": 2,
            "suite_count": 1,
        },
        "game": {
            "move_count": 9,
            "illegal_move_count": 1,
            "replay_refs": ["evidence://artifact/replay"],
        },
        "custom": {"alpha": "beta", "large": {"ignored": True}},
    }

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert "Agent profile" in visible_html
    assert "External harness profile" in visible_html
    assert "Game profile" in visible_html
    assert "custom" in visible_html
    assert "Alpha" in visible_html
    assert "beta" in visible_html
    assert "ignored" not in visible_html


@pytest.mark.fast
def test_static_renderer_hides_zero_signal_scenario_profiles() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["scenario_profiles"] = {
        "agent": {
            "profile_version": "gage.scenario.agent.v1",
            "trial_count": 0,
            "failed_trial_count": 0,
            "representative_ref_ids": [],
        },
        "external_harness": {
            "profile_version": "gage.scenario.external_harness.v1",
            "harnesses": [],
            "trial_count": 0,
            "trial_rollup": {},
        },
        "game": {
            "profile_version": "gage.scenario.game.v1",
            "game_kits": [],
            "illegal_actions": {"games": 0, "total": 0},
            "move_count": 0,
            "replay_refs": [],
        },
    }

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert "Scenario profile" not in visible_html
    assert "Agent profile" not in visible_html
    assert "External harness profile" not in visible_html
    assert "Game profile" not in visible_html


@pytest.mark.fast
def test_static_renderer_does_not_embed_media_data_urls() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["evidence_refs"] = [
        {
            "ref_id": "evidence://media/abc",
            "kind": "media",
            "path": "external://sha256/" + ("a" * 64),
            "mime_type": "image/png",
            "sha256": "a" * 64,
            "preview": {"image_url": "data:image/png;base64,SECRET"},
        }
    ]

    html = StaticReportRenderer().render(context)

    assert "Evidence explorer" in html
    assert "data:image" not in html
    assert "SECRET" not in html
    assert "external://sha256/" in html


@pytest.mark.fast
def test_static_renderer_shows_safe_media_preview_source() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["evidence_refs"] = [
        {
            "ref_id": "evidence://media/abc",
            "kind": "media",
            "path": "external://sha256/" + ("a" * 64),
            "mime_type": "image/png",
            "sha256": "a" * 64,
            "preview": {
                "source": "https://huggingface.co/datasets/MMMU/example.png?token=SECRET#frag",
            },
        }
    ]

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert "https://huggingface.co/datasets/MMMU/example.png" in visible_html
    assert "token=SECRET" not in visible_html
    assert "#frag" not in visible_html
    assert "<pre" in visible_html


@pytest.mark.fast
def test_static_renderer_redacts_embedded_data_urls_in_preview_and_context() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["evidence_refs"] = [
        {
            "ref_id": "evidence://artifact/json-preview",
            "kind": "artifact",
            "path": "artifacts/preview.txt",
            "mime_type": "text/plain",
            "preview": '{"image":"data:image/png;base64,SECRET","caption":"keep"}',
        }
    ]

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert "data:image" not in visible_html
    assert "SECRET" not in visible_html
    assert "data:image" not in html
    assert "SECRET" not in html
    assert "&lt;redacted:data-url&gt;" in visible_html
    assert "\\u003credacted:data-url\\u003e" in html
    assert "caption" in visible_html
    assert "keep" in visible_html


@pytest.mark.fast
def test_static_renderer_redacts_unencoded_svg_data_urls_from_html_and_json() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["evidence_refs"] = [
        {
            "ref_id": "evidence://artifact/svg-preview",
            "kind": "artifact",
            "path": "artifacts/svg.txt",
            "mime_type": "text/plain",
            "preview": (
                'prefix data:image/svg+xml,<svg title="SECRET">'
                "<text>SECRET</text></svg> suffix"
            ),
        }
    ]

    html = StaticReportRenderer().render(context)
    visible_html, embedded_json = html.split(
        '<script type="application/json" id="gage-report-context">',
        1,
    )
    embedded_json = embedded_json.split("</script>", 1)[0]

    for fragment in (visible_html, embedded_json):
        assert "data:image" not in fragment
        assert "<svg" not in fragment
        assert "&lt;svg" not in fragment
        assert "\\u003csvg" not in fragment
        assert "SECRET" not in fragment
    assert "prefix" in visible_html
    assert "suffix" in visible_html


@pytest.mark.fast
def test_static_renderer_allows_long_chips_to_wrap() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["attention_cases"] = [
        {
            "case_id": "case-long-chip",
            "severity": "high",
            "reason_codes": ["reason." + ("very_long_segment_" * 12)],
            "summary": "Long reason code.",
            "scoring": {"priority_score": 0.5},
        }
    ]

    html = StaticReportRenderer().render(context)

    assert 'class="chip mono"' in html
    assert "overflow-wrap: anywhere;" in html
    assert "white-space: normal;" in html


@pytest.mark.fast
def test_static_renderer_includes_planned_badge_state_css() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["headline"]["verdict"] = "passed-with-warnings"
    context["attention_cases"] = [
        {
            "case_id": "case-info",
            "severity": "info",
            "summary": "Informational finding.",
            "scoring": {"priority_score": 0.1},
        }
    ]
    context["failure_clusters"] = [
        {
            "cluster_id": "cluster-degraded",
            "cluster_key": ["runtime.degraded"],
            "count": 1,
            "severity": "info",
            "label": "Degraded cluster",
        }
    ]

    html = StaticReportRenderer().render(context)

    assert 'class="badge verdict verdict-passed-with-warnings"' in html
    assert 'class="badge severity severity-info"' in html
    assert ".verdict-passed-with-warnings" in html
    assert ".verdict-degraded" in html
    assert ".severity-info" in html


@pytest.mark.fast
def test_static_renderer_adds_primary_metric_ratio_bar() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["headline"]["primary_metric"] = {
        "metric_id": "pass_rate",
        "name": "Pass rate",
        "value": 0.42,
        "unit": "ratio",
    }

    html = StaticReportRenderer().render(context)

    assert 'class="metric-bar"' in html
    assert 'class="metric-fill"' in html
    assert 'style="width: 42%;"' in html


def _rich_context() -> dict:
    context = ReportContext.minimal("run-rich").to_dict()
    context["headline"] = {
        **context["headline"],
        "verdict": "failed",
        "verdict_reason": "Critical scheduler failure.",
        "one_line_summary": "Run needs review.",
        "primary_metric": {"metric_id": "pass_rate", "value": 0.42, "unit": "ratio"},
    }
    context["runtime_health"] = {
        "sample_count": 10,
        "completed_count": 7,
        "failed_count": 2,
        "aborted_count": 1,
        "duration_s": 12.345,
    }
    context["observability_health"] = {"events_emitted_total": 99}
    context["metrics"] = [
        {
            "metric_id": "pass_rate",
            "name": "Pass rate",
            "value": 0.42,
            "scope": "run",
            "metadata": {"huge": "x" * 5000},
        }
    ]
    context["attention_cases"] = [
        {
            "case_id": "case-1",
            "severity": "critical",
            "reason_codes": ["scheduler.failed"],
            "summary": "Scheduler stopped.",
            "evidence_ref_ids": ["evidence://artifact/log"],
            "sample_id": "sample-1",
            "trial_id": "trial-1",
            "scoring": {
                "frequency": 0.2,
                "impact": "critical",
                "actionability": "high",
                "priority_score": 0.91,
            },
        }
    ]
    context["case_details"] = {
        "case-1": {
            "message_history_preview": [{"role": "assistant", "content": "hello <world>"}],
            "artifact_preview_ref_ids": ["evidence://artifact/log"],
            "evidence_ref_ids": ["evidence://artifact/log"],
            "truncated": True,
        }
    }
    context["failure_clusters"] = [
        {
            "cluster_id": "cluster-1",
            "cluster_key": ["scheduler.failed"],
            "count": 2,
            "severity": "high",
            "sample_ids": ["sample-1", "sample-2"],
            "representative_ref_ids": ["evidence://artifact/log"],
            "label": "Scheduler failures",
            "hypothesis": "Retry budget exhausted.",
        }
    ]
    context["outliers"] = [
        {
            "metric_id": "latency_s",
            "scope": "run",
            "ranking": "desc",
            "top_k": [
                {
                    "sample_id": "sample-9",
                    "value": 99.9,
                    "evidence_ref_ids": ["evidence://artifact/log"],
                }
            ],
        }
    ]
    context["evidence_refs"] = [
        {
            "ref_id": "evidence://artifact/log",
            "kind": "artifact",
            "path": "artifacts/log.json",
            "mime_type": "application/json",
            "size_bytes": 2048,
            "sha256": "b" * 64,
            "timestamp_iso": "2026-05-15T00:00:00Z",
            "sample_id": "sample-1",
            "task_id": "task-1",
            "preview": {
                "text": "hello <world>\n" + ("A" * 2400),
                "metadata": {"nested": True},
            },
        }
    ]
    return context
