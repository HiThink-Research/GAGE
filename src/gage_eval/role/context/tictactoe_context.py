"""Tic-Tac-Toe context provider for arena games."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.registry import registry
from gage_eval.role.arena.games.gomoku.coord_scheme import GomokuCoordCodec, normalize_coord_scheme
from gage_eval.role.arena.games.tictactoe.env import DEFAULT_PLAYER_IDS, TicTacToeLocalCore


@registry.asset(
    "context_impls",
    "tictactoe_context",
    desc="Tic-Tac-Toe rules and board context provider",
    tags=("tictactoe", "context"),
)
class TicTacToeContext:
    """Injects Tic-Tac-Toe rules and initial board state into the sample."""

    def __init__(
        self,
        *,
        board_size: int = 3,
        start_player: str = DEFAULT_PLAYER_IDS[0],
        coord_scheme: str = "ROW_COL",
    ) -> None:
        self._board_size = board_size
        self._start_player = start_player
        self._coord_scheme = coord_scheme

    def provide(self, payload: Dict[str, Any], _state=None) -> Dict[str, Any]:
        """Inject a Tic-Tac-Toe system prompt and board snapshot."""

        sample = payload.get("sample") or {}
        params = payload.get("params") or {}
        metadata = sample.get("metadata") or {}

        board_size = int(metadata.get("board_size", params.get("board_size", self._board_size)))
        coord_scheme = normalize_coord_scheme(
            metadata.get("coord_scheme", params.get("coord_scheme", self._coord_scheme))
        )
        player_ids = metadata.get("player_ids") or list(DEFAULT_PLAYER_IDS)
        player_ids = [str(player_id) for player_id in player_ids]
        player_names = metadata.get("player_names") or {}
        if isinstance(player_names, list):
            player_names = {
                player_ids[idx]: name for idx, name in enumerate(player_names) if idx < len(player_ids)
            }
        if not isinstance(player_names, dict):
            player_names = {}
        for player_id in player_ids:
            player_names.setdefault(player_id, player_id)
        start_player_id = metadata.get("start_player_id") or params.get("start_player_id", self._start_player)
        if start_player_id not in player_ids:
            for player_id, name in player_names.items():
                if name == start_player_id:
                    start_player_id = player_id
                    break
        if start_player_id not in player_ids:
            start_player_id = player_ids[0] if player_ids else DEFAULT_PLAYER_IDS[0]

        core = TicTacToeLocalCore(
            board_size=board_size,
            start_player=start_player_id,
            coord_scheme=coord_scheme,
        )
        board_text = core.render_board()

        coord_codec = GomokuCoordCodec(board_size=board_size, coord_scheme=coord_scheme)
        column_labels = coord_codec.column_labels()
        column_range = f"{column_labels[0]}-{column_labels[-1]}" if column_labels else "-"
        row_range = f"1-{board_size}"
        example_coord = coord_codec.index_to_coord(0, 0)

        ordered_names = [player_names.get(player_id, player_id) for player_id in player_ids]
        players_hint = " vs ".join(ordered_names) if ordered_names else "two players"
        start_player_name = player_names.get(start_player_id, start_player_id)
        system_text = (
            "You are a Tic-Tac-Toe (3x3) player.\n"
            "Your Objective:\n"
            "- Win by forming a straight line of three marks horizontally, vertically, or diagonally.\n"
            "- Prevent your opponent from doing the same.\n"
            "\n"
            "Rules:\n"
            f"- Players: {players_hint}.\n"
            f"- {start_player_name} moves first. Players alternate turns.\n"
            "- Place one mark per turn on an empty cell.\n"
            f"- Board size: {board_size}x{board_size}.\n"
            "\n"
            "Coordinates:\n"
            f"- Columns: {column_range} (Left to Right).\n"
            f"- Rows: {row_range} (Bottom to Top).\n"
            f"- Example: '{example_coord}' is the bottom-left corner.\n"
            "\n"
            "Board Representation:\n"
            "- The first line is the column header.\n"
            "- Each subsequent line starts with a row number followed by the cell values for that row.\n"
            f"- Rows are listed from top (row {board_size}) to bottom (row 1).\n"
            "- Row numbers increase upward (row 1 is the bottom row).\n"
            "- Empty cells are shown as '.'; players are shown as X and O by default.\n"
            "\n"
            "Output Requirements:\n"
            f"- Output your move as a single coordinate (e.g., '{example_coord}').\n"
            "- The final line must be ONLY the coordinate.\n"
        )
        user_text = (
            f"Game start. {start_player_name} moves first.\n"
            "Board:\n"
            f"{board_text}"
        )

        _ensure_messages(sample, system_text, user_text)
        sample["metadata"] = metadata
        return {"context": {"system": system_text, "board": board_text}}


def _ensure_messages(sample: Dict[str, Any], system_text: str, user_text: str) -> None:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        messages = []

    if not messages or messages[0].get("role") != "system":
        messages.insert(
            0,
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}],
            },
        )
    messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
    sample["messages"] = messages
