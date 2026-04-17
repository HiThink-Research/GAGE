from __future__ import annotations

from gage_eval.role.judge.appworld_evaluate import _build_appworld_diagnostics


def test_appworld_diagnostics_surface_failed_assertions() -> None:
    diagnostic_reason, diagnostic_details = _build_appworld_diagnostics(
        {
            "task_id": "task-1",
            "tests": {
                "passes": [{"label": "no_op_pass"}],
                "fails": [
                    {
                        "label": "no_op_fail",
                        "requirement": "assert answers match.",
                        "trace": "AssertionError: '<<not_given>>' == 'a love that never was'",
                    }
                ],
            },
        }
    )

    assert diagnostic_reason == "verifier_assertion_failed"
    assert diagnostic_details == {
        "verifier_failures": [
            {
                "label": "no_op_fail",
                "requirement": "assert answers match.",
                "trace_excerpt": "AssertionError: '<<not_given>>' == 'a love that never was'",
            }
        ]
    }
