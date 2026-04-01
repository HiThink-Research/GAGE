from __future__ import annotations

from gage_eval.game_kits.aec_env_game.pettingzoo.visualization import (
    VISUALIZATION_SPEC as PETTINGZOO_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.real_time_game.retro_platformer.visualization import (
    VISUALIZATION_SPEC as RETRO_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.real_time_game.openra.visualization import (
    VISUALIZATION_SPEC as OPENRA_VISUALIZATION_SPEC,
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
                "board_text": "Tick 17. Legal actions: 0, 1, 2, 3, 4",
                "active_player": "p0",
                "metadata": {
                    "reward": 0.75,
                },
                "view": {
                    "text": "Tick 17. Legal actions: 0, 1, 2, 3, 4",
                },
                "legal_actions": {
                    "items": ["0", "1", "2", "3", "4"],
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
        {"id": "3", "label": "3", "text": "3"},
        {"id": "4", "label": "4", "text": "4"},
    )
    assert {"kind": "badge", "label": "Stream", "value": "pov"} in scene.overlays


def test_vizdoom_frame_projection_normalizes_numeric_legal_action_items() -> None:
    scene = assemble_visual_scene(
        visual_session=VisualSession(
            session_id="vizdoom-live-sample",
            game_id="vizdoom",
            plugin_id=VIZDOOM_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="p0", observer_kind="player"),
        ),
        event=TimelineEvent(
            seq=18,
            ts_ms=3018,
            type="snapshot",
            label="snapshot",
        ),
        snapshot_body={
            "tick": 18,
            "step": 18,
            "actor": "p0",
            "observation": {
                "board_text": "Tick 18. Legal actions: 0, 1, 2, 3, 4",
                "active_player": "p0",
                "metadata": {
                    "action_mapping": {
                        "0": "NOOP",
                        "1": "ATTACK",
                        "2": "TURN_LEFT",
                        "3": "TURN_RIGHT",
                        "4": "MOVE_FORWARD",
                    }
                },
                "legal_actions": {
                    "items": [0, 1, 2, 3, 4],
                },
                "context": {
                    "mode": "tick",
                    "tick": 18,
                    "step": 18,
                },
            },
        },
        visualization_spec=VIZDOOM_VISUALIZATION_SPEC,
    )

    assert scene.legal_actions == (
        {"id": "0", "label": "No-op", "text": "No-op"},
        {"id": "1", "label": "Fire", "text": "Fire"},
        {"id": "2", "label": "Turn Left", "text": "Turn Left"},
        {"id": "3", "label": "Turn Right", "text": "Turn Right"},
        {"id": "4", "label": "Move Forward", "text": "Move Forward"},
    )


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


def test_openra_rts_projection_exposes_units_economy_objectives_and_selection() -> None:
    scene = assemble_visual_scene(
        visual_session=VisualSession(
            session_id="openra-sample",
            game_id="openra",
            plugin_id=OPENRA_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="player_0", observer_kind="player"),
        ),
        event=TimelineEvent(
            seq=31,
            ts_ms=5031,
            type="snapshot",
            label="snapshot",
        ),
        snapshot_body={
            "tick": 31,
            "step": 6,
            "move_count": 6,
            "last_move": "queue_production:barracks:ranger",
            "observation": {
                "board_text": "Credits 1200 | Power +10 | Objective hold the ridge",
                "active_player": "player_0",
                "metadata": {
                    "map_id": "ra_map01",
                    "map": {
                        "id": "ra_map01",
                        "mod_id": "ra",
                        "title": "Marigold Town",
                        "map_size": {"width": 99, "height": 99},
                        "bounds": {"x": 1, "y": 1, "width": 97, "height": 97},
                        "image_size": {"width": 97, "height": 97},
                        "preview_source": "reference_map_preview",
                    },
                    "economy": {
                        "credits": 1200,
                        "income_per_minute": 320,
                        "power": {"produced": 100, "used": 90},
                    },
                    "objectives": [
                        {"id": "hold_ridge", "label": "Hold the ridge", "status": "active"},
                        {"id": "destroy_radar", "label": "Destroy radar dome", "status": "pending"},
                    ],
                    "selection": {
                        "unit_ids": ["mcv_1", "rifle_2"],
                        "primary_unit_id": "mcv_1",
                    },
                    "units": [
                        {
                            "id": "mcv_1",
                            "owner": "player_0",
                            "label": "MCV",
                            "kind": "vehicle",
                            "hp": 95,
                            "status": "idle",
                            "position": {"x": 12, "y": 4},
                            "selected": True,
                        },
                        {
                            "id": "rifle_2",
                            "owner": "player_0",
                            "label": "Rifle Infantry",
                            "kind": "infantry",
                            "hp": 82,
                            "status": "moving",
                            "position": {"x": 14, "y": 7},
                            "selected": True,
                        },
                    ],
                    "production": {
                        "queues": [
                            {
                                "building_id": "barracks_1",
                                "label": "Barracks",
                                "items": [{"id": "ranger", "label": "Ranger", "progress": 0.4}],
                            }
                        ]
                    },
                },
                "legal_actions": {
                    "items": [
                        {
                            "id": "select_units",
                            "label": "Select units",
                            "text": "Select units",
                            "payloadSchema": {"unit_ids": ["<unit-id>", "<unit-id>"]},
                        },
                        {
                            "id": "issue_command",
                            "label": "Issue command",
                            "text": "Issue command",
                            "payloadSchema": {"command": "attack_move", "target": {"x": 18, "y": 11}},
                        },
                    ]
                },
                "view": {
                    "text": "Credits 1200 | Power +10 | Objective hold the ridge",
                },
                "context": {
                    "mode": "tick",
                    "tick": 31,
                    "step": 6,
                },
            },
            "media": {
                "primary": {
                    "mediaId": "openra-frame-31",
                    "transport": "artifact_ref",
                    "mimeType": "image/png",
                    "url": "frames/openra-frame-31.png",
                }
            },
            "viewport": {
                "width": 1280,
                "height": 720,
            },
        },
        visualization_spec=OPENRA_VISUALIZATION_SPEC,
    )

    assert scene.kind == "rts"
    assert scene.summary["streamId"] == "main"
    assert scene.summary["frameTitle"] == "OpenRA RTS"
    assert scene.body["rts"]["map"] == {
        "id": "ra_map01",
        "modId": "ra",
        "title": "Marigold Town",
        "gridSize": {"width": 99, "height": 99},
        "bounds": {"x": 1, "y": 1, "width": 97, "height": 97},
        "imageSize": {"width": 97, "height": 97},
        "previewSource": "reference_map_preview",
    }
    assert scene.body["rts"]["selection"] == {
        "unitIds": ["mcv_1", "rifle_2"],
        "primaryUnitId": "mcv_1",
    }
    assert scene.body["rts"]["economy"] == {
        "credits": 1200,
        "incomePerMinute": 320,
        "power": {"produced": 100, "used": 90},
    }
    assert scene.body["rts"]["objectives"] == (
        {"id": "hold_ridge", "label": "Hold the ridge", "status": "active"},
        {"id": "destroy_radar", "label": "Destroy radar dome", "status": "pending"},
    )
    assert scene.body["rts"]["units"][0] == {
        "id": "mcv_1",
        "owner": "player_0",
        "label": "MCV",
        "kind": "vehicle",
        "hp": 95,
        "status": "idle",
        "position": {"x": 12, "y": 4},
        "selected": True,
    }
    assert scene.body["status"] == {
        "activePlayerId": "player_0",
        "observerPlayerId": "player_0",
        "tick": 31,
        "step": 6,
        "moveCount": 6,
        "lastMove": "queue_production:barracks:ranger",
        "reward": None,
    }
    assert scene.legal_actions == (
        {
            "id": "select_units",
            "label": "Select units",
            "text": "Select units",
            "payloadSchema": {"unit_ids": ["<unit-id>", "<unit-id>"]},
        },
        {
            "id": "issue_command",
            "label": "Issue command",
            "text": "Issue command",
            "payloadSchema": {"command": "attack_move", "target": {"x": 18, "y": 11}},
        },
    )
    assert {"kind": "badge", "label": "Credits", "value": "1200"} in scene.overlays


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
