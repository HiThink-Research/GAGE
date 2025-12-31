"""Grid game dataset preprocessor."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.role.arena.games.gomoku.env import DEFAULT_PLAYER_IDS, PLAYER_BLACK


class GridGamePreprocessor(BasePreprocessor):
    """Standardize grid game records into the Sample schema."""

    name = "grid_game_preprocessor"

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sample = dict(record)
        metadata = dict(sample.get("metadata") or {})
        metadata.pop("players", None)
        metadata.pop("start_player", None)

        board_size = int(metadata.get("board_size", kwargs.get("board_size", 15)))
        win_len = int(metadata.get("win_len", kwargs.get("win_len", 5)))
        coord_scheme = metadata.get("coord_scheme", kwargs.get("coord_scheme", "A1"))
        rule_profile = metadata.get("rule_profile", kwargs.get("rule_profile", "freestyle"))
        win_directions = metadata.get("win_directions", kwargs.get("win_directions"))
        player_ids = metadata.get("player_ids") or list(DEFAULT_PLAYER_IDS)
        player_names = metadata.get("player_names") or {}
        if isinstance(player_ids, dict):
            player_ids = list(player_ids.values())
        player_ids = [str(player_id) for player_id in player_ids]
        if isinstance(player_names, list):
            player_names = {player_ids[idx]: name for idx, name in enumerate(player_names) if idx < len(player_ids)}
        if not isinstance(player_names, dict):
            player_names = {}
        for player_id in player_ids:
            player_names.setdefault(player_id, player_id)

        start_player_id = metadata.get("start_player_id")
        if start_player_id not in player_ids:
            for player_id, name in player_names.items():
                if name == start_player_id:
                    start_player_id = player_id
                    break
        if start_player_id not in player_ids:
            start_player_id = player_ids[0] if player_ids else PLAYER_BLACK

        metadata.update(
            {
                "board_size": board_size,
                "win_len": win_len,
                "coord_scheme": coord_scheme,
                "rule_profile": rule_profile,
                "win_directions": list(win_directions) if isinstance(win_directions, list) else win_directions,
                "player_ids": list(player_ids),
                "player_names": dict(player_names),
                "start_player_id": start_player_id,
            }
        )

        sample.setdefault("schema_version", "gage.v1")
        sample.setdefault("task_type", "agent")
        sample.setdefault("messages", [])
        sample.setdefault("choices", [])
        sample.setdefault("references", [])
        sample.setdefault("data_tag", {})
        sample.setdefault("eval_config", {})
        sample.setdefault("inputs", {})
        sample["metadata"] = metadata
        return sample
