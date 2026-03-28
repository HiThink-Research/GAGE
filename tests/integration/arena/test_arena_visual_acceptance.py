from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gage_eval.game_kits.contracts import GameVisualizationSpec
from gage_eval.game_kits.aec_env_game.pettingzoo.visualization import (
    VISUALIZATION_SPEC as PETTINGZOO_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.board_game.gomoku.visualization import (
    VISUALIZATION_SPEC as GOMOKU_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.board_game.tictactoe.visualization import (
    VISUALIZATION_SPEC as TICTACTOE_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.phase_card_game.doudizhu.visualization import (
    VISUALIZATION_SPEC as DOUDIZHU_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.phase_card_game.mahjong.visualization import (
    VISUALIZATION_SPEC as MAHJONG_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.real_time_game.retro_platformer.visualization import (
    VISUALIZATION_SPEC as RETRO_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.real_time_game.vizdoom.visualization import (
    VISUALIZATION_SPEC as VIZDOOM_VISUALIZATION_SPEC,
)
from gage_eval.role.arena.visualization.assembly import assemble_visual_scene
from gage_eval.role.arena.visualization.contracts import ObserverRef, TimelineEvent, VisualScene, VisualSession
from gage_eval.role.arena.visualization.gateway_service import ArenaVisualGatewayQueryService
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder


def _event(seq: int) -> TimelineEvent:
    return TimelineEvent(seq=seq, ts_ms=1000 + seq, type="snapshot", label="snapshot")


def _build_gomoku_scene() -> VisualScene:
    return assemble_visual_scene(
        visual_session=VisualSession(
            session_id="gomoku-acceptance",
            game_id="gomoku",
            plugin_id=GOMOKU_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="Black", observer_kind="player"),
        ),
        event=_event(5),
        snapshot_body={
            "active_player_id": "White",
            "observer_player_id": "Black",
            "board_text": "   A B C\n 3 . . .\n 2 . W .\n 1 B . .",
            "legal_moves": ["B1", "C1", "A3"],
            "move_count": 2,
            "last_move": "B2",
            "player_ids": ["Black", "White"],
            "player_names": {"Black": "Black", "White": "White"},
            "coord_scheme": "A1",
            "winning_line": ["A1", "B2"],
        },
        visualization_spec=GOMOKU_VISUALIZATION_SPEC,
    )


def _build_tictactoe_scene() -> VisualScene:
    return assemble_visual_scene(
        visual_session=VisualSession(
            session_id="tictactoe-acceptance",
            game_id="tictactoe",
            plugin_id=TICTACTOE_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="player_0", observer_kind="player"),
        ),
        event=_event(3),
        snapshot_body={
            "active_player_id": "player_1",
            "observer_player_id": "player_0",
            "board_text": "   1 2 3\n 3 . . .\n 2 . O .\n 1 X . .",
            "legal_moves": ["1,2", "1,3", "3,1"],
            "move_count": 2,
            "last_move": "2,2",
            "player_ids": ["player_0", "player_1"],
            "player_names": {"player_0": "Alpha", "player_1": "Beta"},
            "coord_scheme": "ROW_COL",
            "winning_line": ["1,1", "2,2", "3,3"],
        },
        visualization_spec=TICTACTOE_VISUALIZATION_SPEC,
    )


def _build_doudizhu_scene(observer: ObserverRef) -> VisualScene:
    return assemble_visual_scene(
        visual_session=VisualSession(
            session_id="doudizhu-acceptance",
            game_id="doudizhu",
            plugin_id=DOUDIZHU_VISUALIZATION_SPEC.plugin_id,
            observer=observer,
        ),
        event=_event(7),
        snapshot_body={
            "active_player_id": "player_0",
            "observer_player_id": "player_0",
            "player_ids": ["player_0", "player_1", "player_2"],
            "player_names": {
                "player_0": "Player 0",
                "player_1": "Player 1",
                "player_2": "Player 2",
            },
            "public_state": {
                "landlord_id": "player_0",
                "num_cards_left": {"player_0": 2, "player_1": 2, "player_2": 2},
                "played_cards": [
                    {"player_id": "player_0", "cards": ["3"]},
                    {"player_id": "player_1", "cards": []},
                    {"player_id": "player_2", "cards": []},
                ],
                "seen_cards": ["3"],
                "trace": [{"player": 0, "action": "3"}],
            },
            "private_state": {
                "self_id": "player_0",
                "current_hand": ["3", "4"],
                "current_hand_text": "3, 4",
            },
            "ui_state": {
                "roles": {
                    "player_0": "landlord",
                    "player_1": "peasant",
                    "player_2": "peasant",
                },
                "seat_order": {
                    "bottom": "player_0",
                    "left": "player_1",
                    "right": "player_2",
                },
                "latest_actions": [["3"], [], []],
                "move_history": [{"player_idx": 0, "move": "3"}],
            },
            "legal_moves": ["pass", "4"],
            "chat_log": [{"player_id": "player_1", "text": "watch this"}],
            "move_count": 1,
            "last_move": "3",
        },
        visualization_spec=DOUDIZHU_VISUALIZATION_SPEC,
    )


def _build_mahjong_scene(observer: ObserverRef) -> VisualScene:
    return assemble_visual_scene(
        visual_session=VisualSession(
            session_id="mahjong-acceptance",
            game_id="mahjong",
            plugin_id=MAHJONG_VISUALIZATION_SPEC.plugin_id,
            observer=observer,
        ),
        event=_event(9),
        snapshot_body={
            "active_player_id": "east",
            "observer_player_id": "east",
            "player_ids": ["east", "south", "west", "north"],
            "player_names": {
                "east": "East",
                "south": "South",
                "west": "West",
                "north": "North",
            },
            "public_state": {
                "discards": ["B1", "C1", "D1"],
                "melds": {"south": ["Pong C3"]},
            },
            "private_state": {
                "hand": ["B1", "Red"],
                "hand_raw": ["bamboo-1", "dragons-red"],
            },
            "legal_moves": ["B1", "Red"],
            "chat_log": [{"player_id": "south", "text": "pon"}],
            "move_count": 4,
            "last_move": "C1",
        },
        visualization_spec=MAHJONG_VISUALIZATION_SPEC,
    )


def _build_pettingzoo_scene() -> VisualScene:
    return assemble_visual_scene(
        visual_session=VisualSession(
            session_id="pettingzoo-acceptance",
            game_id="pettingzoo",
            plugin_id=PETTINGZOO_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="pilot_0", observer_kind="player"),
        ),
        event=_event(11),
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


def _build_vizdoom_scene() -> VisualScene:
    return assemble_visual_scene(
        visual_session=VisualSession(
            session_id="vizdoom-acceptance",
            game_id="vizdoom",
            plugin_id=VIZDOOM_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="p0", observer_kind="player"),
        ),
        event=_event(17),
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


def _build_retro_scene() -> VisualScene:
    return assemble_visual_scene(
        visual_session=VisualSession(
            session_id="retro-acceptance",
            game_id="retro_platformer",
            plugin_id=RETRO_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="player_0", observer_kind="player"),
        ),
        event=_event(23),
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


def _assert_non_blank_board(scene: VisualScene) -> None:
    board = scene.body["board"]
    cells = board["cells"]
    assert len(cells) >= 9
    assert any(cell["occupant"] for cell in cells)
    assert any(cell["isLegalAction"] for cell in cells)


def _assert_non_blank_table(scene: VisualScene) -> None:
    table = scene.body["table"]
    seats = table["seats"]
    assert len(seats) >= 3
    assert any(seat["hand"]["isVisible"] or seat["hand"]["maskedCount"] > 0 for seat in seats)
    assert table["center"]["label"]


def _assert_non_blank_frame(scene: VisualScene) -> None:
    assert scene.media is not None
    assert scene.media.primary is not None
    assert scene.media.primary.media_id
    assert scene.body["frame"]["title"]
    assert scene.body["view"]["text"] or scene.body["frame"]["viewport"] is not None


def _gomoku_snapshot_initial() -> dict[str, Any]:
    return {
        "active_player_id": "White",
        "observer_player_id": "White",
        "board_text": "   A B C\n 3 . . .\n 2 . W .\n 1 B . .",
        "legal_moves": ["B1", "C1", "A3"],
        "move_count": 2,
        "last_move": "B2",
        "player_ids": ["Black", "White"],
        "player_names": {"Black": "Black", "White": "White"},
        "coord_scheme": "A1",
        "winning_line": ["A1", "B2"],
    }


def _gomoku_snapshot_updated() -> dict[str, Any]:
    return {
        "active_player_id": "Black",
        "observer_player_id": "White",
        "board_text": "   A B C\n 3 . . .\n 2 . W .\n 1 B W .",
        "legal_moves": ["C1", "A2"],
        "move_count": 3,
        "last_move": "B1",
        "player_ids": ["Black", "White"],
        "player_names": {"Black": "Black", "White": "White"},
        "coord_scheme": "A1",
        "winning_line": ["A1", "B1", "B2"],
    }


def _tictactoe_snapshot_initial() -> dict[str, Any]:
    return {
        "active_player_id": "player_1",
        "observer_player_id": "player_1",
        "board_text": "   1 2 3\n 3 . . .\n 2 . O .\n 1 X . .",
        "legal_moves": ["1,2", "1,3", "3,1"],
        "move_count": 2,
        "last_move": "2,2",
        "player_ids": ["player_0", "player_1"],
        "player_names": {"player_0": "Alpha", "player_1": "Beta"},
        "coord_scheme": "ROW_COL",
        "winning_line": ["1,1", "2,2", "3,3"],
    }


def _tictactoe_snapshot_updated() -> dict[str, Any]:
    return {
        "active_player_id": "player_0",
        "observer_player_id": "player_1",
        "board_text": "   1 2 3\n 3 . . .\n 2 X O .\n 1 X . .",
        "legal_moves": ["1,3", "3,1", "3,2"],
        "move_count": 3,
        "last_move": "1,2",
        "player_ids": ["player_0", "player_1"],
        "player_names": {"player_0": "Alpha", "player_1": "Beta"},
        "coord_scheme": "ROW_COL",
        "winning_line": ["1,1", "1,2", "2,2"],
    }


def _doudizhu_snapshot_initial() -> dict[str, Any]:
    return {
        "active_player_id": "player_0",
        "observer_player_id": "player_2",
        "player_ids": ["player_0", "player_1", "player_2"],
        "player_names": {
            "player_0": "Player 0",
            "player_1": "Player 1",
            "player_2": "Player 2",
        },
        "public_state": {
            "landlord_id": "player_0",
            "num_cards_left": {"player_0": 2, "player_1": 2, "player_2": 2},
            "played_cards": [
                {"player_id": "player_0", "cards": ["3"]},
                {"player_id": "player_1", "cards": []},
                {"player_id": "player_2", "cards": []},
            ],
            "seen_cards": ["3"],
            "trace": [{"player": 0, "action": "3"}],
        },
        "private_state": {
            "self_id": "player_0",
            "current_hand": ["3", "4"],
            "current_hand_text": "3, 4",
        },
        "ui_state": {
            "roles": {
                "player_0": "landlord",
                "player_1": "peasant",
                "player_2": "peasant",
            },
            "seat_order": {
                "bottom": "player_0",
                "left": "player_1",
                "right": "player_2",
            },
            "latest_actions": [["3"], [], []],
            "move_history": [{"player_idx": 0, "move": "3"}],
        },
        "legal_moves": ["pass", "4"],
        "chat_log": [{"player_id": "player_1", "text": "watch this"}],
        "move_count": 1,
        "last_move": "3",
    }


def _doudizhu_snapshot_updated() -> dict[str, Any]:
    return {
        "active_player_id": "player_1",
        "observer_player_id": "player_2",
        "player_ids": ["player_0", "player_1", "player_2"],
        "player_names": {
            "player_0": "Player 0",
            "player_1": "Player 1",
            "player_2": "Player 2",
        },
        "public_state": {
            "landlord_id": "player_0",
            "num_cards_left": {"player_0": 1, "player_1": 2, "player_2": 2},
            "played_cards": [
                {"player_id": "player_0", "cards": ["3", "4"]},
                {"player_id": "player_1", "cards": []},
                {"player_id": "player_2", "cards": []},
            ],
            "seen_cards": ["3", "4"],
            "trace": [{"player": 0, "action": "3"}, {"player": 0, "action": "4"}],
        },
        "private_state": {
            "self_id": "player_0",
            "current_hand": ["5"],
            "current_hand_text": "5",
        },
        "ui_state": {
            "roles": {
                "player_0": "landlord",
                "player_1": "peasant",
                "player_2": "peasant",
            },
            "seat_order": {
                "bottom": "player_0",
                "left": "player_1",
                "right": "player_2",
            },
            "latest_actions": [["4"], [], []],
            "move_history": [{"player_idx": 0, "move": "3"}, {"player_idx": 0, "move": "4"}],
        },
        "legal_moves": ["pass", "5"],
        "chat_log": [{"player_id": "player_2", "text": "your turn"}],
        "move_count": 2,
        "last_move": "4",
    }


def _mahjong_snapshot_initial() -> dict[str, Any]:
    return {
        "active_player_id": "east",
        "observer_player_id": "east",
        "player_ids": ["east", "south", "west", "north"],
        "player_names": {
            "east": "East",
            "south": "South",
            "west": "West",
            "north": "North",
        },
        "public_state": {
            "discards": ["B1", "C1", "D1"],
            "melds": {"south": ["Pong C3"]},
        },
        "private_state": {
            "hand": ["B1", "Red"],
            "hand_raw": ["bamboo-1", "dragons-red"],
        },
        "legal_moves": ["B1", "Red"],
        "chat_log": [{"player_id": "south", "text": "pon"}],
        "move_count": 4,
        "last_move": "C1",
    }


def _mahjong_snapshot_updated() -> dict[str, Any]:
    return {
        "active_player_id": "south",
        "observer_player_id": "east",
        "player_ids": ["east", "south", "west", "north"],
        "player_names": {
            "east": "East",
            "south": "South",
            "west": "West",
            "north": "North",
        },
        "public_state": {
            "discards": ["B1", "C1", "D1", "Red"],
            "melds": {"south": ["Pong C3"]},
        },
        "private_state": {
            "hand": ["B1"],
            "hand_raw": ["bamboo-1"],
        },
        "legal_moves": ["B1"],
        "chat_log": [{"player_id": "east", "text": "discard"}],
        "move_count": 5,
        "last_move": "Red",
    }


def _pettingzoo_snapshot_initial() -> dict[str, Any]:
    return {
        "observer_player_id": "pilot_1",
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


def _pettingzoo_snapshot_updated() -> dict[str, Any]:
    return {
        "observer_player_id": "pilot_1",
        "step": 4,
        "tick": 4,
        "move_count": 4,
        "stream_id": "arena",
        "last_move": "LEFT",
        "observation": {
            "board_text": "Space Invaders wave 4",
            "active_player": "pilot_0",
            "legal_moves": ["NOOP", "LEFT", "RIGHT"],
            "metadata": {
                "env_id": "pettingzoo.atari.space_invaders_v2",
                "player_id": "pilot_0",
                "reward": 2.0,
            },
            "view": {
                "text": "Space Invaders wave 4",
            },
            "context": {
                "mode": "turn",
                "step": 4,
            },
        },
        "media": {
            "primary": {
                "mediaId": "pz-frame-4",
                "transport": "artifact_ref",
                "mimeType": "image/png",
                "url": "frames/pz-frame-4.png",
            }
        },
    }


def _vizdoom_snapshot_initial() -> dict[str, Any]:
    return {
        "observer_player_id": "broadcast_cam",
        "tick": 17,
        "step": 17,
        "stream_id": "pov",
        "actor": "p0",
        "last_move": "FIRE",
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
    }


def _vizdoom_snapshot_updated() -> dict[str, Any]:
    return {
        "observer_player_id": "broadcast_cam",
        "tick": 18,
        "step": 18,
        "stream_id": "pov",
        "actor": "p0",
        "last_move": "MOVE_FORWARD",
        "observation": {
            "board_text": "Tick 18. Legal actions: 1, 2, 3",
            "active_player": "p0",
            "metadata": {
                "reward": 1.0,
            },
            "view": {
                "text": "Tick 18. Legal actions: 1, 2, 3",
            },
            "legal_actions": {
                "items": ["1", "2", "3"],
            },
            "context": {
                "mode": "tick",
                "tick": 18,
                "step": 18,
            },
        },
        "media": {
            "primary": {
                "mediaId": "vizdoom-frame-18",
                "transport": "artifact_ref",
                "mimeType": "image/jpeg",
                "url": "frames/vizdoom-frame-18.jpg",
            }
        },
        "viewport": {
            "width": 320,
            "height": 180,
        },
    }


def _retro_snapshot_initial() -> dict[str, Any]:
    return {
        "observer_player_id": "broadcast_cam",
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
    }


def _retro_snapshot_updated() -> dict[str, Any]:
    return {
        "observer_player_id": "broadcast_cam",
        "tick": 24,
        "step": 6,
        "move_count": 6,
        "last_move": "right",
        "reward": 3.0,
        "observation": {
            "board_text": "Mario x03 · coins 18",
            "active_player": "player_0",
            "legal_moves": [
                {"id": "noop", "label": "No-op", "text": "No-op"},
                {"id": "right", "label": "Move Right", "text": "Move Right", "hold_ticks": 6},
                {"id": "jump", "label": "Jump", "text": "Jump", "hold_ticks": 4},
            ],
            "metadata": {
                "env_id": "SuperMarioBros3-Nes-v0",
            },
            "view": {
                "text": "Mario x03 · coins 18",
            },
            "context": {
                "mode": "tick",
                "tick": 24,
                "step": 6,
            },
        },
        "media": {
            "primary": {
                "mediaId": "retro-frame-24",
                "transport": "artifact_ref",
                "mimeType": "image/png",
                "url": "frames/retro-frame-24.png",
            }
        },
    }


def _persist_acceptance_artifact(
    tmp_path: Path,
    *,
    game_id: str,
    visualization_spec: GameVisualizationSpec,
    scheduling_family: str,
    active_player_id: str,
    initial_snapshot: dict[str, Any],
    updated_snapshot: dict[str, Any],
    action_payload: dict[str, Any],
) -> Path:
    replay_path = tmp_path / "runs" / game_id / "replays" / f"{game_id}-acceptance" / "replay.json"
    recorder = ArenaVisualSessionRecorder(
        plugin_id=visualization_spec.plugin_id,
        game_id=game_id,
        scheduling_family=scheduling_family,
        session_id=f"{game_id}-acceptance",
        observer_modes=tuple(visualization_spec.observer_schema.get("supported_modes", ())),
    )
    recorder.record_snapshot(
        ts_ms=1001,
        step=1,
        tick=1,
        snapshot=initial_snapshot,
        label="before_intent",
    )
    recorder.record_decision_window_open(
        ts_ms=1002,
        step=1,
        tick=1,
        player_id=active_player_id,
        observation=initial_snapshot,
    )
    recorder.record_action_intent(
        ts_ms=1003,
        step=1,
        tick=1,
        player_id=active_player_id,
        action=action_payload,
        observation=initial_snapshot,
    )
    recorder.record_snapshot(
        ts_ms=1004,
        step=2,
        tick=2,
        snapshot=updated_snapshot,
        label="after_intent",
    )
    return recorder.persist(replay_path).manifest_path


def _assert_scene_contains_action(scene: VisualScene, expected_token: str) -> None:
    expected = str(expected_token)
    assert any(
        expected in {
            str(action.get("id", "")),
            str(action.get("label", "")),
            str(action.get("text", "")),
            str(action.get("coord", "")),
        }
        for action in scene.legal_actions
    )


def _assert_observer_projection(
    primary_scene: VisualScene,
    alternate_scene: VisualScene,
    *,
    expected_primary_observer_player_id: str | None,
    expected_alternate_observer_player_id: str | None,
    private_player_id: str | None = None,
    expected_visible_cards: tuple[str, ...] = (),
) -> None:
    assert primary_scene.body["status"]["observerPlayerId"] == expected_primary_observer_player_id
    assert alternate_scene.body["status"]["observerPlayerId"] == expected_alternate_observer_player_id

    if private_player_id is None:
        return

    primary_seats = {seat["playerId"]: seat for seat in primary_scene.body["table"]["seats"]}
    alternate_seats = {seat["playerId"]: seat for seat in alternate_scene.body["table"]["seats"]}
    assert primary_seats[private_player_id]["hand"]["isVisible"] is True
    assert primary_seats[private_player_id]["hand"]["cards"] == list(expected_visible_cards)
    assert alternate_seats[private_player_id]["hand"]["isVisible"] is False
    assert alternate_seats[private_player_id]["hand"]["cards"] == []


@dataclass(frozen=True)
class AcceptanceCase:
    game_id: str
    visualization_spec: GameVisualizationSpec
    scheduling_family: str
    initial_snapshot: Callable[[], dict[str, Any]]
    updated_snapshot: Callable[[], dict[str, Any]]
    active_player_id: str
    expected_action_token: str
    primary_observer: ObserverRef
    alternate_observer: ObserverRef
    surface_assertion: Callable[[VisualScene], None]
    expected_pre_last_move: str
    expected_post_last_move: str
    expected_primary_observer_player_id: str | None
    expected_alternate_observer_player_id: str | None
    private_player_id: str | None = None
    expected_visible_cards: tuple[str, ...] = ()


_ACCEPTANCE_CASES: tuple[AcceptanceCase, ...] = (
    AcceptanceCase(
        game_id="gomoku",
        visualization_spec=GOMOKU_VISUALIZATION_SPEC,
        scheduling_family="turn",
        initial_snapshot=_gomoku_snapshot_initial,
        updated_snapshot=_gomoku_snapshot_updated,
        active_player_id="White",
        expected_action_token="B1",
        primary_observer=ObserverRef(observer_id="Black", observer_kind="player"),
        alternate_observer=ObserverRef(observer_id="", observer_kind="global"),
        surface_assertion=_assert_non_blank_board,
        expected_pre_last_move="B2",
        expected_post_last_move="B1",
        expected_primary_observer_player_id="Black",
        expected_alternate_observer_player_id="White",
    ),
    AcceptanceCase(
        game_id="tictactoe",
        visualization_spec=TICTACTOE_VISUALIZATION_SPEC,
        scheduling_family="turn",
        initial_snapshot=_tictactoe_snapshot_initial,
        updated_snapshot=_tictactoe_snapshot_updated,
        active_player_id="player_1",
        expected_action_token="1,2",
        primary_observer=ObserverRef(observer_id="player_0", observer_kind="player"),
        alternate_observer=ObserverRef(observer_id="", observer_kind="global"),
        surface_assertion=_assert_non_blank_board,
        expected_pre_last_move="2,2",
        expected_post_last_move="1,2",
        expected_primary_observer_player_id="player_0",
        expected_alternate_observer_player_id="player_1",
    ),
    AcceptanceCase(
        game_id="doudizhu",
        visualization_spec=DOUDIZHU_VISUALIZATION_SPEC,
        scheduling_family="turn",
        initial_snapshot=_doudizhu_snapshot_initial,
        updated_snapshot=_doudizhu_snapshot_updated,
        active_player_id="player_0",
        expected_action_token="4",
        primary_observer=ObserverRef(observer_id="player_0", observer_kind="player"),
        alternate_observer=ObserverRef(observer_id="", observer_kind="global"),
        surface_assertion=_assert_non_blank_table,
        expected_pre_last_move="3",
        expected_post_last_move="4",
        expected_primary_observer_player_id="player_0",
        expected_alternate_observer_player_id=None,
        private_player_id="player_0",
        expected_visible_cards=("5",),
    ),
    AcceptanceCase(
        game_id="mahjong",
        visualization_spec=MAHJONG_VISUALIZATION_SPEC,
        scheduling_family="turn",
        initial_snapshot=_mahjong_snapshot_initial,
        updated_snapshot=_mahjong_snapshot_updated,
        active_player_id="east",
        expected_action_token="Red",
        primary_observer=ObserverRef(observer_id="east", observer_kind="player"),
        alternate_observer=ObserverRef(observer_id="", observer_kind="global"),
        surface_assertion=_assert_non_blank_table,
        expected_pre_last_move="C1",
        expected_post_last_move="Red",
        expected_primary_observer_player_id="east",
        expected_alternate_observer_player_id=None,
        private_player_id="east",
        expected_visible_cards=("B1",),
    ),
    AcceptanceCase(
        game_id="pettingzoo",
        visualization_spec=PETTINGZOO_VISUALIZATION_SPEC,
        scheduling_family="agent_cycle",
        initial_snapshot=_pettingzoo_snapshot_initial,
        updated_snapshot=_pettingzoo_snapshot_updated,
        active_player_id="pilot_0",
        expected_action_token="FIRE",
        primary_observer=ObserverRef(observer_id="pilot_0", observer_kind="player"),
        alternate_observer=ObserverRef(observer_id="", observer_kind="global"),
        surface_assertion=_assert_non_blank_frame,
        expected_pre_last_move="FIRE",
        expected_post_last_move="LEFT",
        expected_primary_observer_player_id="pilot_0",
        expected_alternate_observer_player_id="pilot_1",
    ),
    AcceptanceCase(
        game_id="vizdoom",
        visualization_spec=VIZDOOM_VISUALIZATION_SPEC,
        scheduling_family="real_time_tick",
        initial_snapshot=_vizdoom_snapshot_initial,
        updated_snapshot=_vizdoom_snapshot_updated,
        active_player_id="p0",
        expected_action_token="2",
        primary_observer=ObserverRef(observer_id="p0", observer_kind="player"),
        alternate_observer=ObserverRef(observer_id="", observer_kind="camera"),
        surface_assertion=_assert_non_blank_frame,
        expected_pre_last_move="FIRE",
        expected_post_last_move="MOVE_FORWARD",
        expected_primary_observer_player_id="p0",
        expected_alternate_observer_player_id="broadcast_cam",
    ),
    AcceptanceCase(
        game_id="retro_platformer",
        visualization_spec=RETRO_VISUALIZATION_SPEC,
        scheduling_family="real_time_tick",
        initial_snapshot=_retro_snapshot_initial,
        updated_snapshot=_retro_snapshot_updated,
        active_player_id="player_0",
        expected_action_token="right_jump",
        primary_observer=ObserverRef(observer_id="player_0", observer_kind="player"),
        alternate_observer=ObserverRef(observer_id="", observer_kind="camera"),
        surface_assertion=_assert_non_blank_frame,
        expected_pre_last_move="right_jump",
        expected_post_last_move="right",
        expected_primary_observer_player_id="player_0",
        expected_alternate_observer_player_id="broadcast_cam",
    ),
)


@pytest.mark.parametrize("case", _ACCEPTANCE_CASES, ids=lambda case: case.game_id)
def test_arena_visual_acceptance_matrix_exercises_render_playback_seek_observer_and_human_intent(
    tmp_path: Path,
    case: AcceptanceCase,
) -> None:
    manifest_path = _persist_acceptance_artifact(
        tmp_path,
        game_id=case.game_id,
        visualization_spec=case.visualization_spec,
        scheduling_family=case.scheduling_family,
        active_player_id=case.active_player_id,
        initial_snapshot=case.initial_snapshot(),
        updated_snapshot=case.updated_snapshot(),
        action_payload={"move": case.expected_action_token},
    )
    query_service = ArenaVisualGatewayQueryService(
        visualization_spec=case.visualization_spec,
    )
    session = query_service.load_session(manifest_path)
    timeline_page = query_service.page_timeline(manifest_path, limit=10)
    decision_window_page = query_service.page_timeline(manifest_path, after_seq=1, limit=2)
    intent_scene = query_service.load_scene(
        manifest_path,
        seq=3,
        observer=case.primary_observer,
    )
    pre_seek_scene = query_service.load_scene(
        manifest_path,
        seq=1,
        observer=case.primary_observer,
    )
    post_seek_scene = query_service.load_scene(
        manifest_path,
        seq=4,
        observer=case.primary_observer,
    )
    alternate_observer_scene = query_service.load_scene(
        manifest_path,
        seq=4,
        observer=case.alternate_observer,
    )

    assert session.game_id == case.game_id
    assert session.plugin_id == case.visualization_spec.plugin_id
    assert session.playback.can_seek is True
    assert session.capabilities == {
        "supportsReplay": True,
        "supportsTimeline": True,
        "supportsSeek": True,
        "observerModes": list(case.visualization_spec.observer_schema["supported_modes"]),
    }
    assert [event.type for event in timeline_page.events] == [
        "snapshot",
        "decision_window_open",
        "action_intent",
        "snapshot",
    ]
    assert [event.type for event in decision_window_page.events] == [
        "decision_window_open",
        "action_intent",
    ]
    assert decision_window_page.has_more is True

    assert intent_scene is not None
    assert pre_seek_scene is not None
    assert post_seek_scene is not None
    assert alternate_observer_scene is not None

    _assert_scene_contains_action(intent_scene, case.expected_action_token)
    case.surface_assertion(pre_seek_scene)
    case.surface_assertion(post_seek_scene)

    assert pre_seek_scene.scene_id.endswith(":seq:1")
    assert post_seek_scene.scene_id.endswith(":seq:4")
    assert pre_seek_scene.body["status"]["lastMove"] == case.expected_pre_last_move
    assert post_seek_scene.body["status"]["lastMove"] == case.expected_post_last_move
    _assert_observer_projection(
        post_seek_scene,
        alternate_observer_scene,
        expected_primary_observer_player_id=case.expected_primary_observer_player_id,
        expected_alternate_observer_player_id=case.expected_alternate_observer_player_id,
        private_player_id=case.private_player_id,
        expected_visible_cards=case.expected_visible_cards,
    )


@pytest.mark.parametrize(
    ("scene_builder", "player_id", "expected_visible_cards"),
    [
        (_build_doudizhu_scene, "player_0", ["3", "4"]),
        (_build_mahjong_scene, "east", ["B1", "Red"]),
    ],
)
def test_arena_visual_acceptance_table_observer_switch_reprojects_private_view(
    scene_builder: Callable[[ObserverRef], VisualScene],
    player_id: str,
    expected_visible_cards: list[str],
) -> None:
    spectator_scene = scene_builder(ObserverRef(observer_id="", observer_kind="spectator"))
    player_scene = scene_builder(ObserverRef(observer_id=player_id, observer_kind="player"))

    spectator_seats = {
        seat["playerId"]: seat
        for seat in spectator_scene.body["table"]["seats"]
    }
    player_seats = {
        seat["playerId"]: seat
        for seat in player_scene.body["table"]["seats"]
    }

    assert spectator_scene.body["status"]["observerPlayerId"] is None
    assert spectator_seats[player_id]["hand"]["isVisible"] is False
    assert spectator_seats[player_id]["hand"]["cards"] == []

    assert player_scene.body["status"]["observerPlayerId"] == player_id
    assert player_seats[player_id]["hand"]["isVisible"] is True
    assert player_seats[player_id]["hand"]["cards"] == expected_visible_cards
