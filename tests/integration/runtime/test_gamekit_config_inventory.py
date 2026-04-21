from __future__ import annotations

import pytest

from gage_eval.tools.gamekit_acceptance import materialize_gamekit_config
from tests._support.gamekit_matrix import (
    HEADLESS_NO_HUMAN,
    HUMAN_VISUAL,
    LIVE_CASES_BY_CATEGORY,
    PLUGIN_IDS_BY_GAME_KIT,
    VISUAL_NO_HUMAN,
    discover_live_gamekit_configs,
    expected_live_gamekit_config_paths,
    human_visual_families,
    human_visual_required_families,
    iter_live_cases,
    load_primary_adapter_params,
    shipped_gamekit_families,
)


@pytest.mark.fast
def test_live_gamekit_config_inventory_matches_frozen_matrix() -> None:
    assert len(LIVE_CASES_BY_CATEGORY[HEADLESS_NO_HUMAN]) == 26
    assert len(LIVE_CASES_BY_CATEGORY[VISUAL_NO_HUMAN]) == 31
    assert len(LIVE_CASES_BY_CATEGORY[HUMAN_VISUAL]) == 19

    assert discover_live_gamekit_configs() == expected_live_gamekit_config_paths()


@pytest.mark.fast
@pytest.mark.parametrize("case", iter_live_cases(category=HEADLESS_NO_HUMAN), ids=lambda case: case.relpath)
def test_headless_no_human_configs_keep_headless_runtime_semantics(case) -> None:
    params = load_primary_adapter_params(case)
    visualizer = params.get("visualizer") or {}
    human_input = params.get("human_input") or {}

    assert params["game_kit"] == case.game_kit
    assert params["env"] == case.env
    assert human_input.get("enabled") is not True
    assert visualizer.get("enabled") is not True
    assert visualizer.get("mode") != "arena_visual"


@pytest.mark.fast
@pytest.mark.parametrize("case", iter_live_cases(category=VISUAL_NO_HUMAN), ids=lambda case: case.relpath)
def test_visual_no_human_configs_keep_visual_runtime_semantics(case) -> None:
    params = load_primary_adapter_params(case)
    visualizer = params.get("visualizer") or {}
    human_input = params.get("human_input") or {}

    assert params["game_kit"] == case.game_kit
    assert params["env"] == case.env
    assert human_input.get("enabled") is not True
    assert visualizer["enabled"] is True
    assert visualizer["launch_browser"] is True
    assert visualizer["mode"] == "arena_visual"
    assert case.plugin_id == PLUGIN_IDS_BY_GAME_KIT[case.game_kit]
    if case.live_scene_scheme is not None:
        assert visualizer["live_scene_scheme"] == case.live_scene_scheme
    else:
        assert "live_scene_scheme" not in visualizer


@pytest.mark.fast
@pytest.mark.parametrize("case", iter_live_cases(category=HUMAN_VISUAL), ids=lambda case: case.relpath)
def test_human_visual_configs_keep_human_and_visual_runtime_semantics(case) -> None:
    params = load_primary_adapter_params(case)
    visualizer = params.get("visualizer") or {}
    human_input = params.get("human_input") or {}

    assert params["game_kit"] == case.game_kit
    assert params["env"] == case.env
    assert human_input["enabled"] is True
    assert visualizer["enabled"] is True
    assert visualizer["launch_browser"] is True
    assert visualizer["mode"] == "arena_visual"
    assert case.plugin_id == PLUGIN_IDS_BY_GAME_KIT[case.game_kit]


@pytest.mark.fast
def test_every_shipped_gamekit_family_has_human_visual_coverage() -> None:
    assert human_visual_families() == human_visual_required_families()
    assert "pettingzoo" in shipped_gamekit_families()


@pytest.mark.io
@pytest.mark.parametrize("case", iter_live_cases(), ids=lambda case: case.relpath)
def test_live_gamekit_configs_materialize_runtime(case, tmp_path) -> None:
    materialize_gamekit_config(
        case.path,
        output_dir=tmp_path / "runs",
        run_id=case.path.stem,
        gpus=0,
        cpus=1,
    )
