from gage_eval.role.arena.games.retro.action_codec import RetroActionCodec
from gage_eval.role.arena.games.retro.retro_env import StableRetroArenaEnvironment


def test_retro_env_observation_includes_controls_summary_and_move_aliases():
    env = StableRetroArenaEnvironment(
        game="SuperMarioBros3-Nes-v0",
        display_mode="headless",
        legal_moves=["noop", "right_run_jump", "select", "start"],
    )
    env._action_codec = RetroActionCodec(  # type: ignore[attr-defined]
        buttons=["LEFT", "RIGHT", "UP", "DOWN", "A", "B", "SELECT", "START"],
        legal_moves=["noop", "right_run_jump", "select", "start"],
    )

    obs = env.observe("player_0")

    controls = (obs.metadata.get("observation_extra") or {}).get("controls")
    assert isinstance(controls, dict)
    assert controls.get("scheme") == "wasd_jkl"
    assert "W/A/S/D" in str(controls.get("keys_hint"))

    move_aliases = controls.get("move_aliases")
    assert isinstance(move_aliases, dict)
    assert move_aliases["right_run_jump"]["keys_combo"] == "d+j+k"
    assert move_aliases["select"]["keys_combo"] == "l"
    assert move_aliases["start"]["keys_combo"] == "enter"

    assert "Controls:" in (obs.view_text or "")
    assert "Stable-retro buttons:" in (obs.view_text or "")
    assert "right_run_jump=d+j+k (RIGHT+A+B)" in (obs.view_text or "")
