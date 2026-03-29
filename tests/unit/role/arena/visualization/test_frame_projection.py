from __future__ import annotations

from gage_eval.game_kits.aec_env_game.pettingzoo.visualization import (
    VISUALIZATION_SPEC as PETTINGZOO_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.real_time_game.retro_platformer.visualization import (
    VISUALIZATION_SPEC as RETRO_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.real_time_game.vizdoom.visualization import (
    VISUALIZATION_SPEC as VIZDOOM_VISUALIZATION_SPEC,
)
from gage_eval.role.arena.visualization.assembly import assemble_visual_scene
from gage_eval.role.arena.visualization.contracts import ObserverRef, TimelineEvent, VisualSession


def test_pettingzoo_frame_projection_builds_visible_frame_surface_contract() -> None:
    scene = assemble_visual_scene(
        visual_session=VisualSession(
            session_id="pettingzoo-sample",
            game_id="pettingzoo",
            plugin_id=PETTINGZOO_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="pilot_0", observer_kind="player"),
        ),
        event=TimelineEvent(
            seq=11,
            ts_ms=2011,
            type="snapshot",
            label="snapshot",
        ),
        snapshot_body={
            "step": 3,
            "tick": 3,
            "move_count": 3,
            "stream_id": "arena",
            "last_move": "FIRE",
            "observation": {
                "board_text": "Space Invaders wave 3",
                "active_player": "pilot_0",
                "legal_moves": ["NOOP", "LEFT", "FIRE"],
                "metadata": {
                    "env_id": "pettingzoo.atari.space_invaders_v2",
                    "player_id": "pilot_0",
                    "reward": 1.5,
                },
                "view": {
                    "text": "Space Invaders wave 3",
                },
                "context": {
                    "mode": "turn",
                    "step": 3,
                },
            },
            "media": {
                "primary": {
                    "mediaId": "pz-frame-3",
                    "transport": "artifact_ref",
                    "mimeType": "image/png",
                    "url": "frames/pz-frame-3.png",
                }
            },
        },
        visualization_spec=PETTINGZOO_VISUALIZATION_SPEC,
    )

    assert scene.kind == "frame"
    assert scene.summary["streamId"] == "arena"
    assert scene.summary["frameTitle"] == "PettingZoo Frame"
    assert scene.media is not None
    assert scene.media.primary is not None
    assert scene.media.primary.media_id == "pz-frame-3"
    assert scene.body["frame"] == {
        "title": "PettingZoo Frame",
        "subtitle": "Stream arena",
        "altText": "PettingZoo frame",
        "streamId": "arena",
        "fit": "contain",
        "viewport": None,
    }
    assert scene.body["status"] == {
        "activePlayerId": "pilot_0",
        "observerPlayerId": "pilot_0",
        "tick": 3,
        "step": 3,
        "moveCount": 3,
        "lastMove": "FIRE",
        "reward": 1.5,
    }
    assert scene.body["view"]["text"] == "Space Invaders wave 3"
    assert scene.legal_actions == (
        {"id": "NOOP", "label": "NOOP", "text": "NOOP"},
        {"id": "LEFT", "label": "LEFT", "text": "LEFT"},
        {"id": "FIRE", "label": "FIRE", "text": "FIRE"},
    )
    assert scene.overlays == (
        {"kind": "badge", "label": "Tick", "value": "3"},
        {"kind": "badge", "label": "Reward", "value": "1.5"},
        {"kind": "badge", "label": "Last move", "value": "FIRE"},
    )


def test_vizdoom_frame_projection_exposes_pov_hud_and_viewport() -> None:
    scene = assemble_visual_scene(
        visual_session=VisualSession(
            session_id="vizdoom-sample",
            game_id="vizdoom",
            plugin_id=VIZDOOM_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="p0", observer_kind="player"),
        ),
        event=TimelineEvent(
            seq=17,
            ts_ms=3017,
            type="snapshot",
            label="snapshot",
        ),
        snapshot_body={
            "tick": 17,
            "step": 17,
            "stream_id": "pov",
            "actor": "p0",
            "observation": {
                "board_text": "Tick 17. Legal actions: 0, 1, 2",
                "active_player": "p0",
                "metadata": {
                    "reward": 0.75,
                },
                "view": {
                    "text": "Tick 17. Legal actions: 0, 1, 2",
                },
                "legal_actions": {
                    "items": ["0", "1", "2"],
                },
                "context": {
                    "mode": "tick",
                    "tick": 17,
                    "step": 17,
                },
            },
            "media": {
                "primary": {
                    "mediaId": "vizdoom-frame-17",
                    "transport": "artifact_ref",
                    "mimeType": "image/jpeg",
                    "url": "frames/vizdoom-frame-17.jpg",
                }
            },
            "viewport": {
                "width": 320,
                "height": 180,
            },
        },
        visualization_spec=VIZDOOM_VISUALIZATION_SPEC,
    )

    assert scene.summary["streamId"] == "pov"
    assert scene.body["frame"]["title"] == "ViZDoom Frame"
    assert scene.body["frame"]["viewport"] == {"width": 320, "height": 180}
    assert scene.body["status"]["reward"] == 0.75
    assert scene.body["status"]["observerPlayerId"] == "p0"
    assert scene.legal_actions == (
        {"id": "0", "label": "0", "text": "0"},
        {"id": "1", "label": "1", "text": "1"},
        {"id": "2", "label": "2", "text": "2"},
    )
    assert {"kind": "badge", "label": "Stream", "value": "pov"} in scene.overlays


def test_retro_frame_projection_uses_single_player_status_and_controls() -> None:
    scene = assemble_visual_scene(
        visual_session=VisualSession(
            session_id="retro-sample",
            game_id="retro_platformer",
            plugin_id=RETRO_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="player_0", observer_kind="player"),
        ),
        event=TimelineEvent(
            seq=23,
            ts_ms=4023,
            type="snapshot",
            label="snapshot",
        ),
        snapshot_body={
            "tick": 23,
            "step": 5,
            "move_count": 5,
            "last_move": "right_jump",
            "reward": 2.0,
            "observation": {
                "board_text": "Mario x03 · coins 17",
                "active_player": "player_0",
                "legal_moves": [
                    {"id": "noop", "label": "No-op", "text": "No-op"},
                    {"id": "right", "label": "Move Right", "text": "Move Right", "hold_ticks": 6},
                    {
                        "id": "right_jump",
                        "label": "Right + Jump",
                        "text": "Right + Jump",
                        "hold_ticks": 6,
                    },
                ],
                "metadata": {
                    "env_id": "SuperMarioBros3-Nes-v0",
                },
                "view": {
                    "text": "Mario x03 · coins 17",
                },
                "context": {
                    "mode": "tick",
                    "tick": 23,
                    "step": 5,
                },
            },
            "media": {
                "primary": {
                    "mediaId": "retro-frame-23",
                    "transport": "artifact_ref",
                    "mimeType": "image/png",
                    "url": "frames/retro-frame-23.png",
                }
            },
        },
        visualization_spec=RETRO_VISUALIZATION_SPEC,
    )

    assert scene.summary["frameTitle"] == "Retro Mario Frame"
    assert scene.body["frame"]["subtitle"] == "Tick 23"
    assert scene.body["status"] == {
        "activePlayerId": "player_0",
        "observerPlayerId": "player_0",
        "tick": 23,
        "step": 5,
        "moveCount": 5,
        "lastMove": "right_jump",
        "reward": 2.0,
    }
    assert scene.legal_actions == (
        {"id": "noop", "label": "No-op", "text": "No-op"},
        {"id": "right", "label": "Move Right", "text": "Move Right", "hold_ticks": 6},
        {"id": "right_jump", "label": "Right + Jump", "text": "Right + Jump", "hold_ticks": 6},
    )


def test_pettingzoo_frame_projection_unwraps_snapshot_observation_from_intent_event() -> None:
    snapshot = {
        "step": 3,
        "tick": 3,
        "move_count": 3,
        "stream_id": "arena",
        "last_move": "FIRE",
        "observation": {
            "board_text": "Space Invaders wave 3",
            "active_player": "pilot_0",
            "legal_moves": ["NOOP", "LEFT", "FIRE"],
            "metadata": {
                "env_id": "pettingzoo.atari.space_invaders_v2",
                "player_id": "pilot_0",
                "reward": 1.5,
            },
            "view": {
                "text": "Space Invaders wave 3",
            },
            "context": {
                "mode": "turn",
                "step": 3,
            },
        },
        "media": {
            "primary": {
                "mediaId": "pz-frame-3",
                "transport": "artifact_ref",
                "mimeType": "image/png",
                "url": "frames/pz-frame-3.png",
            }
        },
    }

    scene = assemble_visual_scene(
        visual_session=VisualSession(
            session_id="pettingzoo-sample",
            game_id="pettingzoo",
            plugin_id=PETTINGZOO_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="pilot_0", observer_kind="player"),
        ),
        event=TimelineEvent(
            seq=12,
            ts_ms=2012,
            type="action_intent",
            label="action_intent",
            payload={
                "step": 3,
                "tick": 3,
                "playerId": "pilot_0",
                "action": {"move": "FIRE"},
                # Recorder action_intent events can wrap the full frame snapshot here.
                "observation": snapshot,
            },
        ),
        snapshot_body=snapshot,
        visualization_spec=PETTINGZOO_VISUALIZATION_SPEC,
    )

    assert scene.body["status"]["activePlayerId"] == "pilot_0"
    assert scene.legal_actions == (
        {"id": "NOOP", "label": "NOOP", "text": "NOOP"},
        {"id": "LEFT", "label": "LEFT", "text": "LEFT"},
        {"id": "FIRE", "label": "FIRE", "text": "FIRE"},
    )
