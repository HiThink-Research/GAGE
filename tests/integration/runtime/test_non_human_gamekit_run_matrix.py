from __future__ import annotations

from pathlib import Path
import json

import pytest

from gage_eval.game_kits.real_time_game.openra.native_bridge import OPENRA_BRIDGE_DIR_ENV
from gage_eval.tools.gamekit_acceptance import run_gamekit_config
from tests._support.gamekit_matrix import (
    HEADLESS_NO_HUMAN,
    VISUAL_NO_HUMAN,
    iter_live_cases,
)


NON_HUMAN_CASES = (
    *iter_live_cases(category=HEADLESS_NO_HUMAN),
    *iter_live_cases(category=VISUAL_NO_HUMAN),
)


@pytest.mark.io
@pytest.mark.parametrize("case", NON_HUMAN_CASES, ids=lambda case: case.relpath)
def test_non_human_gamekit_configs_run_end_to_end(case, tmp_path) -> None:
    result = run_gamekit_config(
        case.path,
        output_dir=tmp_path / "runs",
        run_id=case.path.stem,
        gpus=0,
        cpus=1,
        launch_browser=False if case.category == VISUAL_NO_HUMAN else None,
        linger_after_finish_s=0.0 if case.category == VISUAL_NO_HUMAN else None,
    )

    sample = result.sample
    output = result.output
    replay_path = output.get("result", {}).get("replay_path")

    assert output["sample"]["game_kit"] == case.game_kit
    assert output["sample"]["env"] == case.env
    assert output["tick"] > 0
    assert output["step"] > 0
    assert output["result"]["move_count"] > 0
    assert isinstance(output["result"]["result"], str) and output["result"]["result"]
    assert output["arena_trace"]
    assert sample["predict_result"][0]["arena_trace"] == list(output["arena_trace"])
    if replay_path:
        assert Path(str(replay_path)).exists()


@pytest.mark.io
def test_openra_native_dummy_visual_config_replays_non_camera_mouse_actions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_dir = tmp_path / "openra-bridge"
    monkeypatch.setenv(OPENRA_BRIDGE_DIR_ENV, str(bridge_dir))

    result = run_gamekit_config(
        Path("config/custom/openra/openra_ra_skirmish_native_dummy_visual_gamekit.yaml"),
        output_dir=tmp_path / "runs",
        run_id="openra-ra-skirmish-native-dummy-visual-regression",
        gpus=0,
        cpus=1,
        launch_browser=False,
        linger_after_finish_s=0.0,
    )

    input_path = bridge_dir / "input.ndjson"
    assert input_path.exists()
    events = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result.output["sample"]["runtime_overrides"]["native_demo_script"] == "unit_patrol"
    assert any(
        str(event.get("event_type")) == "mouse_down" and str(event.get("button")) == "left"
        for event in events
    )
    assert any(
        str(event.get("event_type")) == "mouse_down" and str(event.get("button")) == "right"
        for event in events
    )
    assert not any(str(event.get("event_type", "")).startswith("key_") for event in events)
