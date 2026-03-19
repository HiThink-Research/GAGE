from __future__ import annotations

import pytest

from gage_eval.role.adapters.arena import ArenaRoleAdapter


@pytest.mark.fast
def test_arena_adapter_resolves_prepared_sample_identity() -> None:
    assert (
        ArenaRoleAdapter._resolve_sample_id(
            {
                "_gage_task_id": "task-alpha",
                "_gage_source_index": 7,
                "_gage_sample_id_source": "synthetic_source_index",
                "sample_id": "task-alpha:7",
            }
        )
        == "task-alpha:7"
    )


@pytest.mark.fast
def test_arena_adapter_builds_fallback_from_ingress_metadata() -> None:
    assert (
        ArenaRoleAdapter._resolve_sample_id(
            {
                "_gage_task_id": "task-alpha",
                "_gage_source_index": 2,
            }
        )
        == "task-alpha:2"
    )
