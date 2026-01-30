"""Local Gomoku engine with board state and rule checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from gage_eval.role.arena.games.gomoku.coord_scheme import GomokuCoordCodec, normalize_coord_scheme
from gage_eval.role.arena.games.gomoku.rules import GomokuRuleEngine
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult
from gage_eval.registry import registry

PLAYER_BLACK = "Black"
PLAYER_WHITE = "White"
DEFAULT_PLAYER_IDS = (PLAYER_BLACK, PLAYER_WHITE)
DEFAULT_TOKEN_MAP = {PLAYER_BLACK: "B", PLAYER_WHITE: "W"}
EMPTY_CELL = "."


@dataclass(frozen=True)
class GomokuMove:
    """Represents a legal move applied to the board."""

    index: int
    player: str
    row: int
    col: int
    coord: str


@dataclass(frozen=True)
class GomokuMoveResult:
    """Describes the outcome of applying a move."""

    is_legal: bool
    winner: Optional[str]
    is_draw: bool
    reason: Optional[str]
    move: Optional[GomokuMove]
    win_direction: Optional[str] = None
    line_length: Optional[int] = None


class GomokuLocalCore:
    """Manages Gomoku board state, validation, and win conditions."""

    def __init__(
        self,
        board_size: int = 15,
        win_len: int = 5,
        start_player: str = PLAYER_BLACK,
        coord_scheme: str = "A1",
        rule_profile: str = "freestyle",
        win_directions: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize the core Gomoku state.

        Args:
            board_size: Board dimension (board_size x board_size).
            win_len: Consecutive stones required to win.
            start_player: Player identifier who moves first.
            coord_scheme: Coordinate scheme identifier.
        """

        if board_size < 1:
            raise ValueError("board_size must be positive")
        if win_len < 2:
            raise ValueError("win_len must be at least 2")
        if win_len > board_size:
            raise ValueError("win_len cannot exceed board_size")
        self.board_size = board_size
        self.win_len = win_len
        self.coord_scheme = normalize_coord_scheme(coord_scheme)
        self._coord_codec = GomokuCoordCodec(board_size=board_size, coord_scheme=self.coord_scheme)
        self.rule_profile = str(rule_profile or "freestyle")
        self.win_directions = tuple(win_directions or ("horizontal", "vertical", "diagonal", "anti_diagonal"))
        self._player_ids = list(DEFAULT_PLAYER_IDS)
        self._player_tokens = dict(DEFAULT_TOKEN_MAP)
        self.start_player = start_player
        self._rule_engine = GomokuRuleEngine(
            win_len=win_len,
            empty_cell=EMPTY_CELL,
            rule_profile=self.rule_profile,
            win_directions=self.win_directions,
        )
        self.reset(start_player=start_player)

    def configure_players(
        self,
        *,
        player_ids: Sequence[str],
        token_map: Optional[Dict[str, str]] = None,
        start_player_id: Optional[str] = None,
    ) -> None:
        """Configure player identifiers and tokens.

        Args:
            player_ids: Ordered player identifiers.
            token_map: Optional mapping from player_id to board token.
            start_player_id: Optional override for the starting player.
        """

        ids = [str(player_id) for player_id in player_ids]
        if len(ids) != 2:
            raise ValueError("Gomoku requires exactly two players")
        tokens = dict(token_map or {})
        defaults = ["B", "W"]
        for idx, player_id in enumerate(ids):
            tokens.setdefault(player_id, defaults[idx] if idx < len(defaults) else str(idx))
        self._player_ids = ids
        self._player_tokens = tokens
        self.start_player = str(start_player_id or ids[0])

    def reset(self, *, start_player: Optional[str] = None) -> None:
        """Reset the board and counters to start a new game.

        Args:
            start_player: Optional override for the next player to move (player_id).
        """

        self.board: List[List[str]] = [
            [EMPTY_CELL for _ in range(self.board_size)] for _ in range(self.board_size)
        ]
        self.active_player = start_player or self.start_player
        self.winner: Optional[str] = None
        self.is_draw = False
        self._game_over = False
        self.move_count = 0
        self.illegal_move_count = 0
        self._move_log: List[GomokuMove] = []
        self._winning_line: Optional[List[Tuple[int, int]]] = None
        self._win_direction: Optional[str] = None
        self._line_length: Optional[int] = None

    def is_terminal(self) -> bool:
        """Return True if the game has ended."""

        return self._game_over

    def legal_moves(self) -> List[Tuple[int, int]]:
        """Return all legal moves as row/col tuples."""

        moves: List[Tuple[int, int]] = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == EMPTY_CELL:
                    moves.append((row, col))
        return moves

    def legal_coords(self) -> List[str]:
        """Return all legal moves as coordinate strings."""

        return [self.index_to_coord(row, col) for row, col in self.legal_moves()]

    def apply_move(self, player: str, row: int, col: int) -> GomokuMoveResult:
        """Apply a move and return the resulting state summary.

        Args:
            player: The player making the move.
            row: Zero-based row index.
            col: Zero-based column index.

        Returns:
            The outcome of applying the move.
        """

        # STEP 1: Validate game state and move ownership
        if self._game_over:
            self.illegal_move_count += 1
            return GomokuMoveResult(False, self.winner, self.is_draw, "game_over", None)
        if player not in self._player_tokens:
            self.illegal_move_count += 1
            return GomokuMoveResult(False, None, False, "unknown_player", None)
        if player != self.active_player:
            self.illegal_move_count += 1
            return GomokuMoveResult(False, None, False, "wrong_player", None)
        if not self._in_bounds(row, col):
            self.illegal_move_count += 1
            return GomokuMoveResult(False, None, False, "out_of_bounds", None)
        if self.board[row][col] != EMPTY_CELL:
            self.illegal_move_count += 1
            return GomokuMoveResult(False, None, False, "occupied", None)

        # STEP 2: Apply the move to the board
        token = self._player_tokens[player]
        self.board[row][col] = token
        self.move_count += 1
        coord = self.index_to_coord(row, col)
        move = GomokuMove(self.move_count, player, row, col, coord)
        self._move_log.append(move)

        # STEP 3: Evaluate terminal conditions
        win_info = self._rule_engine.find_win_info(self.board, row, col)
        if win_info:
            self.winner = player
            self._game_over = True
            self._winning_line = list(win_info.line)
            self._win_direction = win_info.direction
            self._line_length = win_info.line_length
            return GomokuMoveResult(
                True,
                player,
                False,
                "five_in_row",
                move,
                win_direction=win_info.direction,
                line_length=win_info.line_length,
            )
        if self.move_count >= self.board_size * self.board_size:
            self.is_draw = True
            self._game_over = True
            return GomokuMoveResult(True, None, True, "draw", move)

        # STEP 4: Switch active player
        self.active_player = self._other_player(player)
        return GomokuMoveResult(True, None, False, None, move)

    def render_board(self) -> str:
        """Render the board as a plain-text grid for debugging."""

        header = "   " + " ".join(self._column_labels())
        lines = [header]
        for row in range(self.board_size - 1, -1, -1):
            row_label = f"{row + 1:>2}"
            lines.append(f"{row_label} " + " ".join(self.board[row]))
        return "\n".join(lines)

    def format_move_log(self) -> str:
        """Return a readable move log for debugging."""

        return "\n".join(f"{move.index}. {move.player} {move.coord}" for move in self._move_log)

    def move_log(self) -> Sequence[GomokuMove]:
        """Return a copy of the move log."""

        return list(self._move_log)

    def winning_line(self) -> Optional[List[Tuple[int, int]]]:
        """Return the winning line coordinates if the game is won."""

        if not self._winning_line:
            return None
        return list(self._winning_line)

    def win_direction(self) -> Optional[str]:
        """Return the winning line direction if the game is won."""

        return self._win_direction

    def line_length(self) -> Optional[int]:
        """Return the winning line length if the game is won."""

        return self._line_length

    def index_to_coord(self, row: int, col: int) -> str:
        """Convert a row/col index into a coordinate string."""

        return self._coord_codec.index_to_coord(row, col)

    def coord_to_index(self, coord: str) -> Tuple[int, int]:
        """Convert a coordinate string into a row/col index."""

        return self._coord_codec.coord_to_index(coord)

    def _column_labels(self) -> List[str]:
        return self._coord_codec.column_labels()

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def _other_player(self, player: str) -> str:
        if len(self._player_ids) != 2:
            raise ValueError("Gomoku requires exactly two players")
        return self._player_ids[1] if player == self._player_ids[0] else self._player_ids[0]


@registry.asset(
    "arena_impls",
    "gomoku_local_v1",
    desc="Local Gomoku environment (15x15, five-in-row)",
    tags=("gomoku", "arena"),
)
class GomokuArenaEnvironment:
    """Arena environment that wraps the local Gomoku core."""

    def __init__(
        self,
        *,
        board_size: int = 15,
        win_len: int = 5,
        player_ids: Optional[Sequence[str]] = None,
        player_names: Optional[Dict[str, str]] = None,
        token_map: Optional[Dict[str, str]] = None,
        start_player_id: Optional[str] = None,
        start_player: Optional[str] = None,
        coord_scheme: str = "A1",
        rule_profile: str = "freestyle",
        win_directions: Optional[Sequence[str]] = None,
        illegal_policy: Optional[Dict[str, str | int]] = None,
    ) -> None:
        resolved_player_ids = [str(pid) for pid in (player_ids or DEFAULT_PLAYER_IDS)]
        if len(resolved_player_ids) != 2:
            raise ValueError("Gomoku requires exactly two players")
        self._board_size = board_size
        self._win_len = win_len
        self._coord_scheme = normalize_coord_scheme(coord_scheme)
        self._rule_profile = str(rule_profile or "freestyle")
        self._win_directions = tuple(win_directions or ("horizontal", "vertical", "diagonal", "anti_diagonal"))
        self._player_ids = resolved_player_ids
        self._player_names = _normalize_player_names(
            player_ids=resolved_player_ids,
            player_names=player_names,
        )
        resolved_start_player = _resolve_start_player_id(
            player_ids=resolved_player_ids,
            player_names=self._player_names,
            start_player_id=start_player_id,
            start_player=start_player,
        )
        self._start_player_id = resolved_start_player
        self._illegal_policy = dict(illegal_policy or {})
        self._max_illegal = int(self._illegal_policy.get("retry", 0))
        self._illegal_on_fail = str(self._illegal_policy.get("on_fail", "loss"))
        self._core = GomokuLocalCore(
            board_size=board_size,
            win_len=win_len,
            start_player=resolved_start_player,
            coord_scheme=self._coord_scheme,
            rule_profile=self._rule_profile,
            win_directions=self._win_directions,
        )
        self._core.configure_players(
            player_ids=resolved_player_ids,
            token_map=token_map,
            start_player_id=resolved_start_player,
        )
        self._illegal_counts = {player_id: 0 for player_id in resolved_player_ids}
        self._last_move: Optional[str] = None
        self._final_result: Optional[GameResult] = None
        self._winning_line_coords: Optional[List[str]] = None
        self._win_direction: Optional[str] = None
        self._line_length: Optional[int] = None

    def reset(self) -> None:
        self._core.reset(start_player=self._start_player_id)
        self._illegal_counts = {player_id: 0 for player_id in self._player_ids}
        self._last_move = None
        self._final_result = None
        self._winning_line_coords = None
        self._win_direction = None
        self._line_length = None

    def get_active_player(self) -> str:
        return self._core.active_player

    def observe(self, player: str) -> ArenaObservation:
        board_text = self._core.render_board()
        legal_moves = self._core.legal_coords()
        view = {"text": board_text}
        legal_actions = {"items": list(legal_moves)}
        context = {"mode": "turn", "step": self._core.move_count}
        return ArenaObservation(
            board_text=board_text,
            legal_moves=legal_moves,
            active_player=self._core.active_player,
            last_move=self._last_move,
            metadata={
                "board_size": self._board_size,
                "win_len": self._win_len,
                "move_count": self._core.move_count,
                "player_id": player,
                "player_ids": list(self._player_ids),
                "player_names": dict(self._player_names),
                "coord_scheme": self._coord_scheme,
                "rule_profile": self._rule_profile,
                "win_directions": list(self._win_directions),
                "win_direction": self._win_direction,
                "line_length": self._line_length,
                "winning_line": list(self._winning_line_coords) if self._winning_line_coords else None,
                "last_move": self._last_move,
            },
            view=view,
            legal_actions=legal_actions,
            context=context,
        )

    def apply(self, action: ArenaAction) -> Optional[GameResult]:
        if self._final_result is not None:
            return self._final_result

        # STEP 1: Validate and apply the move through the core engine.
        try:
            row, col = self._core.coord_to_index(action.move)
        except Exception:
            return self._handle_illegal(action, reason="invalid_format")

        move_result = self._core.apply_move(action.player, row, col)
        if not move_result.is_legal:
            return self._handle_illegal(action, reason=move_result.reason or "illegal_move")

        self._last_move = move_result.move.coord if move_result.move else None

        # STEP 2: Check for terminal conditions.
        if move_result.winner or move_result.is_draw:
            if move_result.winner:
                win_line = self._core.winning_line() or []
                if win_line:
                    self._winning_line_coords = [
                        self._core.index_to_coord(row, col) for row, col in win_line
                    ]
                self._win_direction = self._core.win_direction()
                self._line_length = self._core.line_length()
            self._final_result = self.build_result(
                result="win" if move_result.winner else "draw",
                reason=move_result.reason,
            )
            return self._final_result

        return None

    def is_terminal(self) -> bool:
        return self._final_result is not None or self._core.is_terminal()

    def build_result(self, *, result: str, reason: Optional[str]) -> GameResult:
        if self._final_result is not None:
            return self._final_result
        winner = self._core.winner
        is_draw = self._core.is_draw
        final_board = self._core.render_board()
        move_log = [
            {
                "index": move.index,
                "player": move.player,
                "coord": move.coord,
                "row": move.row,
                "col": move.col,
            }
            for move in self._core.move_log()
        ]
        if winner:
            resolved_result = "win"
            resolved_reason = reason or "five_in_row"
        elif is_draw:
            resolved_result = "draw"
            resolved_reason = reason or "draw"
        else:
            resolved_result = result
            resolved_reason = reason
        return GameResult(
            winner=winner,
            result=resolved_result,
            reason=resolved_reason,
            move_count=self._core.move_count,
            illegal_move_count=self._core.illegal_move_count,
            final_board=final_board,
            move_log=move_log,
            rule_profile=self._rule_profile,
            win_direction=self._win_direction,
            line_length=self._line_length,
        )

    def _handle_illegal(self, action: ArenaAction, *, reason: str) -> Optional[GameResult]:
        player = action.player
        if player in self._illegal_counts:
            self._illegal_counts[player] += 1
        if self._max_illegal < 0:
            return None
        if self._illegal_counts.get(player, 0) <= self._max_illegal:
            return None

        if self._illegal_on_fail == "draw":
            winner = None
            result = "draw"
        else:
            winner = self._other_player(player)
            result = "loss"

        self._final_result = GameResult(
            winner=winner,
            result=result,
            reason=reason,
            move_count=self._core.move_count,
            illegal_move_count=self._core.illegal_move_count,
            final_board=self._core.render_board(),
            move_log=[
                {
                    "index": move.index,
                    "player": move.player,
                    "coord": move.coord,
                    "row": move.row,
                    "col": move.col,
                }
                for move in self._core.move_log()
            ],
            rule_profile=self._rule_profile,
            win_direction=self._win_direction,
            line_length=self._line_length,
        )
        return self._final_result

    def _other_player(self, player: str) -> str:
        if len(self._player_ids) != 2:
            raise ValueError("Gomoku requires exactly two players")
        return self._player_ids[1] if player == self._player_ids[0] else self._player_ids[0]


def _normalize_player_names(
    *,
    player_ids: Sequence[str],
    player_names: Optional[Dict[str, str]],
) -> Dict[str, str]:
    names = dict(player_names or {})
    return {player_id: names.get(player_id, player_id) for player_id in player_ids}


def _resolve_start_player_id(
    *,
    player_ids: Sequence[str],
    player_names: Dict[str, str],
    start_player_id: Optional[str],
    start_player: Optional[str],
) -> str:
    if start_player_id and start_player_id in player_ids:
        return start_player_id
    if start_player and start_player in player_ids:
        return start_player
    if start_player:
        for player_id, name in player_names.items():
            if name == start_player:
                return player_id
    return player_ids[0]
