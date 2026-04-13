from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.config import build_default_registry
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.resource_profile import NodeResource, ResourceProfile


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_config(config_path: Path, *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))

    config = PipelineConfig.from_dict(payload)
    registry = build_default_registry()
    trace = ObservabilityTrace()
    runtime = build_runtime(
        config,
        registry=registry,
        resource_profile=ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)]),
        trace=trace,
    )
    captured: list[dict] = []
    runtime.sample_loop.register_hook(lambda sample, store=captured: store.append(sample))
    runtime.run()

    assert captured
    return captured[0]


@pytest.mark.parametrize(
    ("config_relpath", "expected_sample_id", "expected_player_ids"),
    [
        (
            "config/custom/doudizhu/doudizhu_dummy_gamekit.yaml",
            "doudizhu_match_0001",
            ["landlord", "farmer_left", "farmer_right"],
        ),
        (
            "config/custom/mahjong/mahjong_dummy_gamekit.yaml",
            "mahjong_match_0001",
            ["east", "south", "west", "north"],
        ),
    ],
)
def test_dummy_gamekit_configs_use_matching_fixture_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_relpath: str,
    expected_sample_id: str,
    expected_player_ids: list[str],
) -> None:
    sample = _run_config(REPO_ROOT / config_relpath, monkeypatch=monkeypatch, tmp_path=tmp_path)

    metadata = sample["metadata"]

    assert sample["id"] == expected_sample_id
    assert sample["sample_id"] == expected_sample_id
    assert metadata["player_ids"] == expected_player_ids
    assert "board_size" not in metadata
    assert "win_len" not in metadata
    assert "coord_scheme" not in metadata


def test_doudizhu_dummy_stub_keeps_richer_smoke_hands() -> None:
    from gage_eval.game_kits.phase_card_game.doudizhu.envs.classic_3p import _StubDoudizhuCore

    core = _StubDoudizhuCore()

    observation = core.get_observation(0)

    assert len(observation["current_hand"]) >= 17
    assert sum(observation["num_cards_left"]) >= 51
