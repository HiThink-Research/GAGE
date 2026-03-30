from __future__ import annotations

import pytest

from gage_eval.tools.gamekit_acceptance import verify_visual_gamekit_config
from tests._support.gamekit_matrix import VISUAL_NO_HUMAN, iter_live_cases


@pytest.mark.io
@pytest.mark.network
@pytest.mark.parametrize(
    "case",
    iter_live_cases(category=VISUAL_NO_HUMAN),
    ids=lambda case: case.relpath,
)
def test_visual_gamekit_configs_launch_browser_and_render_live_scenes(case, tmp_path) -> None:
    result = verify_visual_gamekit_config(
        case.path,
        expect_plugin=str(case.plugin_id),
        expect_live_scene_scheme=case.live_scene_scheme,
        output_dir=tmp_path / "runs",
        run_id=case.path.stem,
        gpus=0,
        cpus=1,
    )

    probe = result.browser_probe

    assert probe is not None
    assert result.browser_urls
    assert probe.viewer_url == result.browser_urls[0]
    assert probe.session_payload["pluginId"] == case.plugin_id
    assert probe.first_scene["seq"] != probe.last_scene["seq"]
    if case.live_scene_scheme is not None:
        assert probe.first_media_payload is not None
        assert probe.last_media_payload is not None
        assert probe.first_media_payload["transport"] == case.live_scene_scheme
        assert probe.last_media_payload["transport"] == case.live_scene_scheme
