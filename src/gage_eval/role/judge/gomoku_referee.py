"""Gomoku referee implementation for judge step."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.registry import registry
from gage_eval.role.arena.games.gomoku.env import DEFAULT_PLAYER_IDS, GomokuLocalCore
from gage_eval.role.judge.base import JudgeImplementation


@registry.asset(
    "judge_impls",
    "gomoku_referee",
    desc="Gomoku referee that replays moves and validates results",
    tags=("gomoku", "judge"),
)
class GomokuReferee(JudgeImplementation):
    """Replays a game log to validate the winner and illegal moves."""

    def __init__(self, *, strict_validate: bool = True) -> None:
        """Initialize the referee.

        Args:
            strict_validate: Whether to flag mismatches with declared winners.
        """

        self._strict_validate = strict_validate

    def invoke(self, payload: Dict[str, Any], _state=None) -> Dict[str, Any]:
        """Validate Gomoku results by replaying the game log.

        Args:
            payload: Judge payload containing sample metadata and model output.
            _state: Optional state (unused).

        Returns:
            Judge output containing winner, result, and legality fields.
        """

        sample = payload.get("sample") or {}
        model_output = payload.get("model_output") or {}
        metadata = sample.get("metadata") or {}

        board_size = int(metadata.get("board_size", 15))
        win_len = int(metadata.get("win_len", 5))
        coord_scheme = metadata.get("coord_scheme", "A1")
        rule_profile = metadata.get("rule_profile", "freestyle")
        win_directions = metadata.get("win_directions")
        player_ids = metadata.get("player_ids") or list(DEFAULT_PLAYER_IDS)
        player_ids = [str(player_id) for player_id in player_ids]
        player_names = metadata.get("player_names") or {}
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
            start_player_id = player_ids[0] if player_ids else DEFAULT_PLAYER_IDS[0]

        core = GomokuLocalCore(
            board_size=board_size,
            win_len=win_len,
            start_player=start_player_id,
            coord_scheme=coord_scheme,
            rule_profile=rule_profile,
            win_directions=win_directions,
        )
        core.configure_players(player_ids=player_ids, token_map=metadata.get("token_map"), start_player_id=start_player_id)
        game_log = model_output.get("game_log") or model_output.get("moves") or []

        illegal_reason: Optional[str] = None
        illegal_player: Optional[str] = None

        for entry in game_log:
            if not isinstance(entry, dict):
                illegal_reason = "invalid_log_entry"
                break
            player = entry.get("player")
            coord = entry.get("coord") or entry.get("move")
            if not player or not coord:
                illegal_reason = "missing_move"
                illegal_player = player
                break
            try:
                row, col = core.coord_to_index(str(coord))
            except Exception:
                illegal_reason = "invalid_coord"
                illegal_player = player
                break
            move_result = core.apply_move(str(player), row, col)
            if not move_result.is_legal:
                illegal_reason = move_result.reason or "illegal_move"
                illegal_player = player
                break
            if core.is_terminal():
                break

        winner = core.winner
        result = "draw" if core.is_draw else ("win" if winner else "unknown")
        reason = "five_in_row" if winner else ("draw" if core.is_draw else None)

        if illegal_reason:
            if len(player_ids) == 2:
                if illegal_player == player_ids[0]:
                    winner = player_ids[1]
                elif illegal_player == player_ids[1]:
                    winner = player_ids[0]
                else:
                    winner = player_ids[0]
            else:
                winner = player_ids[0] if player_ids else None
            result = "loss"
            reason = illegal_reason

        judge_payload = {
            "winner": winner,
            "result": result,
            "reason": reason,
            "illegal_move_count": core.illegal_move_count,
            "valid": illegal_reason is None,
            "rule_profile": rule_profile,
            "win_direction": core.win_direction(),
            "line_length": core.line_length(),
        }

        if self._strict_validate:
            declared = model_output.get("winner")
            if declared and declared != winner:
                judge_payload["winner_mismatch"] = True

        return judge_payload
