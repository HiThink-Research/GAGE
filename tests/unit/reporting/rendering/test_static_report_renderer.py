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
def test_static_renderer_surfaces_task_failures_without_attention_cases() -> None:
    context = ReportContext.minimal("task-failed").to_dict()
    context["headline"]["verdict"] = "failed"
    context["headline"]["primary_metric"] = None
    context["runtime_health"] = {
        "sample_count": 0,
        "completed_count": 0,
        "failed_count": 0,
        "aborted_count": 0,
        "task_failed_count": 1,
    }
    context["tasks"] = [
        {
            "task_id": "swebench_pro_ansible_1case",
            "status": "failed",
            "execution": {
                "status": "failed",
                "failed_step": "harbor_result",
                "failure": {
                    "error_type": "ExternalHarnessParseError",
                    "message": "harbor.launcher_failed: Harbor launcher failed",
                },
            },
        }
    ]

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert "Key findings" in visible_html
    assert "Task failures" in visible_html
    assert "swebench_pro_ansible_1case" in visible_html
    assert "harbor_result" in visible_html
    assert "harbor.launcher_failed" in visible_html


@pytest.mark.fast
def test_static_renderer_links_attention_evidence_refs_to_explorer_rows() -> None:
    context = _rich_context()

    html = StaticReportRenderer().render(context)

    assert 'href="#evidence-evidence-artifact-log"' in html
    assert 'id="evidence-evidence-artifact-log"' in html
    assert "Preview artifact refs" in html


@pytest.mark.fast
def test_static_renderer_uses_failure_first_layout_and_sticky_nav() -> None:
    context = _rich_context()
    context["headline"]["verdict"] = "failed"

    html = StaticReportRenderer().render(context)

    assert html.index('id="key-findings"') < html.index('id="quick-stats"')
    assert html.index("</header>") < html.index('<nav class="report-shell report-nav"')
    assert "position: sticky;" in html
    assert "top: 0;" in html


@pytest.mark.fast
def test_static_renderer_links_reason_chips_to_glossary() -> None:
    context = _rich_context()

    html = StaticReportRenderer().render(context)

    assert 'href="#reason-scheduler-failed"' in html
    assert 'id="reason-codes-glossary"' in html
    assert 'id="reason-scheduler-failed"' in html
    assert "Scheduler failed" in html
    assert "调度失败" in html


@pytest.mark.fast
def test_static_renderer_highlights_json_preview_and_supports_dark_mode() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["evidence_refs"] = [
        {
            "ref_id": "evidence://artifact/json-preview",
            "kind": "artifact",
            "path": "artifacts/result.json",
            "mime_type": "application/json",
            "sha256": "a" * 64,
            "preview": {"text": '{"score": 0.5, "passed": false, "label": "demo"}'},
        }
    ]

    html = StaticReportRenderer().render(context)

    assert '<span class="json-key">&quot;score&quot;</span>' in html
    assert '<span class="json-number">0.5</span>' in html
    assert '<span class="json-bool">false</span>' in html
    assert '<span class="json-string">&quot;demo&quot;</span>' in html
    assert "@media (prefers-color-scheme: dark)" in html
    assert "color-scheme: light dark;" in html


@pytest.mark.fast
def test_static_renderer_summarizes_scenario_profiles_by_kind() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["scenario_profiles"] = {
        "agent": {
            "trial_count": 3,
            "failed_trial_count": 1,
            "representative_ref_ids": ["evidence://artifact/abc"],
        },
        "external_harness": {
            "harnesses": [{"harness_id": "harbor", "sample_count": 1, "trial_count": 1}],
            "trial_count": 1,
            "trial_rollup": {"completed": 1},
        },
        "game": {
            "game_kits": ["gomoku"],
            "move_count": 9,
            "illegal_actions": {"games": 1, "total": 2},
            "replay_refs": ["evidence://artifact/replay"],
        },
        "custom": {"alpha": "beta", "large": {"ignored": True}},
    }

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert "Agent profile" in visible_html
    assert "External harness profile" in visible_html
    assert "Game profile" in visible_html
    assert "harbor" in visible_html
    assert "completed=1" in visible_html
    assert "gomoku" in visible_html
    assert "games=1, total=2" in visible_html
    assert "No metadata." not in visible_html
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
    assert "Profiles" not in visible_html


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
def test_static_renderer_embeds_safe_remote_media_thumbnail() -> None:
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

    assert '<img class="media-thumb"' in visible_html
    assert 'src="https://huggingface.co/datasets/MMMU/example.png"' in visible_html
    assert "token=SECRET" not in visible_html
    assert "URL digest" in visible_html


@pytest.mark.fast
def test_static_renderer_embeds_redacted_host_only_media_thumbnail() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["evidence_refs"] = [
        {
            "ref_id": "evidence://media/abc",
            "kind": "media",
            "path": "external://sha256/" + ("a" * 64),
            "mime_type": "image/png",
            "sha256": "a" * 64,
            "preview": {
                "source": "huggingface.co/datasets/MMMU/example.png",
            },
        }
    ]

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert '<img class="media-thumb"' in visible_html
    assert 'src="https://huggingface.co/datasets/MMMU/example.png"' in visible_html


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


@pytest.mark.fast
def test_static_renderer_reads_metric_values_dicts() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["headline"]["primary_metric"] = {
        "metric_id": "harbor_score_mean",
        "name": "Harbor score mean",
        "values": {"mean": "0"},
        "raw_values": {"mean": 0.0},
        "scope": "run",
        "unit": "score",
    }
    context["metrics"] = [context["headline"]["primary_metric"]]

    html = StaticReportRenderer().render(context)

    assert "Harbor score mean" in html
    assert ">0.00000<" in html
    assert 'style="width: 0%;"' in html
    assert "No metrics recorded." not in html


@pytest.mark.fast
def test_static_renderer_keeps_metric_unit_out_of_primary_value() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["headline"]["primary_metric"] = {
        "metric_id": "harbor_score_mean",
        "name": "Harbor score mean",
        "values": {"mean": "0"},
        "raw_values": {"mean": 0.0},
        "scope": "run",
        "unit": "score",
    }

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert "<strong>0.00000</strong>" in visible_html
    assert "<strong>0.00000 score</strong>" not in visible_html
    assert "score · Harbor score mean" in visible_html


@pytest.mark.fast
def test_static_renderer_formats_resolve_rate_primary_without_key_prefix() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["headline"]["primary_metric"] = {
        "metric_id": "swebench_resolve_rate",
        "name": "SWE-bench resolve rate",
        "values": {"resolve_rate": "0.00000"},
        "raw_values": {"resolve_rate": 0.0},
        "scope": "run",
    }
    context["metrics"] = [context["headline"]["primary_metric"]]

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert ">0.00000<" in visible_html
    assert "resolve_rate=0.00000" not in visible_html
    assert "overflow-wrap: anywhere;" in html


@pytest.mark.fast
def test_static_renderer_keeps_attention_priority_column_readable() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["attention_cases"] = [
        {
            "case_id": "swebench/very_long_sample_identifier_that_should_not_squeeze_priority",
            "severity": "critical",
            "reason_codes": ["client_execution.tool_retry_budget_exhausted"],
            "summary": "Scheduler failed.",
            "scoring": {"priority_score": 0.855},
        }
    ]

    html = StaticReportRenderer().render(context)

    assert 'class="findings-table attention-cases-table"' in html
    assert ".attention-cases-table th:nth-child(4)" in html
    assert "min-width: 96px;" in html
    assert "<th>Priority</th>" in html


@pytest.mark.fast
def test_static_renderer_reason_glossary_resolves_legacy_and_harbor_codes() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["attention_cases"] = [
        {
            "case_id": "case-legacy",
            "severity": "high",
            "reason_codes": ["missing_appworld_success_signal", "harbor.trial_exception"],
            "summary": "Legacy reason codes.",
            "scoring": {"priority_score": 0.7},
        }
    ]

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert "Missing AppWorld success signal" in visible_html
    assert "Harbor trial exception" in visible_html
    assert "Unregistered reason code" not in visible_html


@pytest.mark.fast
def test_static_renderer_uses_game_profile_as_hero_context_fallback() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["headline"]["primary_metric"] = None
    context["scenario_profiles"] = {
        "game": {
            "profile_version": "gage.scenario.game.v1",
            "game_kits": ["tictactoe"],
            "move_count": 9,
            "illegal_actions": {"games": 0, "total": 0},
            "replay_refs": ["evidence://artifact/replay"],
        }
    }

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert "Run context" in visible_html
    assert "Game" in visible_html
    assert "tictactoe" in visible_html
    assert "Move count" in visible_html
    assert ">9<" in visible_html


@pytest.mark.fast
def test_static_renderer_hides_empty_primary_metric() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["headline"]["primary_metric"] = {"metric_id": "multi_choice_acc", "values": {}}

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert "<strong></strong>" not in visible_html
    assert "Primary metric" not in visible_html


@pytest.mark.fast
def test_static_renderer_formats_acc_value_without_key_prefix() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["metrics"] = [
        {"metric_id": "multi_choice_acc", "name": "", "values": {"acc": "1.00000"}, "raw_values": {"acc": 1.0}},
    ]

    html = StaticReportRenderer().render(context)

    assert ">1.00000<" in html
    assert "acc=1.00000" not in html
    assert "Multi Choice Acc" in html


@pytest.mark.fast
def test_static_renderer_left_aligns_metric_values() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["metrics"] = [
        {
            "metric_id": "sample_completion_rate",
            "name": "Sample completion rate",
            "values": {"rate": "0.00000"},
            "raw_values": {"rate": 0.0},
            "unit": "ratio",
        }
    ]

    html = StaticReportRenderer().render(context)

    assert '<td class="metric-value">0.00000</td>' in html
    assert ".metric-value { text-align: left;" in html


@pytest.mark.fast
def test_static_renderer_links_artifact_paths_and_caps_evidence_identity_columns() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["run"]["run_dir"] = "runs/live"
    context["evidence_refs"] = [
        {
            "ref_id": "evidence://artifact/log",
            "kind": "artifact",
            "path": "artifacts/task/sample/trials/trial_0001/infra/trial_result.json",
            "mime_type": "application/json",
            "size_bytes": 42,
            "sha256": "a" * 64,
            "sample_id": "manual_swebench:instance_future-architect__vuls-" + ("a" * 40),
            "task_id": "manual_swebench_pro_docker_lmstudio_smoke11",
        }
    ]

    html = StaticReportRenderer().render(context)

    assert 'href="../artifacts/task/sample/trials/trial_0001/infra/trial_result.json"' in html
    assert 'class="evidence-path"' in html
    assert 'class="evidence-sample"' in html
    assert 'class="evidence-task"' in html
    assert ".evidence-sample," in html
    assert "max-width: 200px;" in html


@pytest.mark.fast
def test_static_renderer_links_replay_paths() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["evidence_refs"] = [
        {
            "ref_id": "evidence://artifact/replay",
            "kind": "artifact",
            "path": "replays/tictactoe_0001/replay.json",
            "mime_type": "application/json",
        },
        {
            "ref_id": "evidence://artifact/manifest",
            "kind": "artifact",
            "path": "replays/tictactoe_0001/arena_visual_session/v1/manifest.json",
            "mime_type": "application/json",
        },
    ]

    html = StaticReportRenderer().render(context)

    assert 'href="../replays/tictactoe_0001/replay.json"' in html
    assert 'href="../replays/tictactoe_0001/arena_visual_session/v1/manifest.json"' in html


@pytest.mark.fast
def test_static_renderer_wraps_long_identifiers_and_titles_owner_cells() -> None:
    context = ReportContext.minimal("run").to_dict()
    long_owner = "manual_swebench_pro_docker_lmstudio_smoke11:instance_future-architect__vuls-" + ("a" * 44)
    context["metrics"] = [
        {
            "metric_id": "swebench_failure_reason",
            "values": {"client_execution.tool_retry_budget_exhausted": "1.00000"},
            "task_id": long_owner,
        }
    ]
    context["run"]["run_dir"] = "/Users/panke/project/gage-eval-main/runs/live-20260515-175517-t11_harbor_terminal_bench-final-rerun"

    html = StaticReportRenderer().render(context)

    assert "overflow-wrap: anywhere;" in html
    assert ".metric-owner" in html
    assert f'title="{long_owner}"' in html
    assert "word-break: break-all;" in html


@pytest.mark.fast
def test_static_renderer_uses_wall_runtime_timing_for_duration() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["run"]["timings"] = {"wall_runtime_s": 91.2}

    html = StaticReportRenderer().render(context)

    assert "1m 31s" in html


@pytest.mark.fast
def test_static_renderer_defaults_footer_diagnostics_and_reason_glossary_open() -> None:
    context = _rich_context()
    context["diagnostics"] = {"report_pack_status": "completed", "warnings": [], "errors": []}

    html = StaticReportRenderer().render(context)

    assert '<details open>\n      <summary>Diagnostics</summary>' in html
    assert '<details id="reason-codes-glossary" open>' in html


@pytest.mark.fast
def test_static_renderer_aggregates_routine_redaction_warnings() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["diagnostics"] = {
        "report_pack_status": "completed",
        "warnings": [
            {"code": "report_pack.secret_redacted", "path": "report_context.md", "finding_count": 1},
            {"code": "report_pack.secret_redacted", "path": "report.html", "finding_count": 2},
        ],
        "errors": [],
    }

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert "Privacy Redactions" in visible_html
    assert "2 rendered assets" in visible_html
    assert "<h3>Warnings</h3>" not in visible_html
    assert "report_context.md" not in visible_html
    assert "report.html" not in visible_html


@pytest.mark.fast
def test_static_renderer_relativizes_run_dir_and_summarizes_methodology_metadata() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["run"]["run_dir"] = "/Users/panke/project/gage-eval-main/runs/live-e2e/run-1"
    context["methodology"]["run_metadata"] = {
        "active_sink": {"kind": "file"},
        "events_emitted_total": 8,
    }

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert "runs/live-e2e/run-1" in visible_html
    assert "/Users/panke/project" not in visible_html
    assert "Started" not in visible_html
    assert "Finished" not in visible_html
    assert "Duration" in visible_html
    assert "2 fields; see report_context.json" in visible_html
    assert "active_sink, events_emitted_total" not in visible_html


@pytest.mark.fast
def test_static_renderer_relativizes_local_paths_inside_previews() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["evidence_refs"] = [
        {
            "ref_id": "evidence://artifact/preview",
            "kind": "artifact",
            "path": "artifacts/log.json",
            "mime_type": "application/json",
            "preview": {
                "text": (
                    "/Users/panke/project/gage-eval-main/runs/live/run-1/"
                    "external_harness/jobs/result.json"
                )
            },
        }
    ]

    html = StaticReportRenderer().render(context)
    visible_html = html.split('<script type="application/json" id="gage-report-context">', 1)[0]

    assert "runs/live/run-1/external_harness/jobs/result.json" in visible_html
    assert "/Users/panke/project" not in html


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
