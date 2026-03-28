import type { BoardCell, BoardPlayer } from "./boardScene";

interface BoardGridProps {
  gameLabel: string;
  actorLabel: string;
  boardSize: number;
  cells: BoardCell[];
  players: BoardPlayer[];
  legalCoords: Set<string>;
  canSubmitMoves: boolean;
  onSubmitMove: (coord: string) => void;
}

function resolveCellToken(cell: BoardCell, tokenByPlayerId: Map<string, string>): string {
  if (cell.occupant) {
    return cell.occupant;
  }
  if (!cell.playerId) {
    return "";
  }
  return tokenByPlayerId.get(cell.playerId) ?? "";
}

export function BoardGrid({
  gameLabel,
  actorLabel,
  boardSize,
  cells,
  players,
  legalCoords,
  canSubmitMoves,
  onSubmitMove,
}: BoardGridProps) {
  const orderedCells = [...cells].sort((left, right) => {
    if (left.row !== right.row) {
      return left.row - right.row;
    }
    return left.col - right.col;
  });
  const tokenByPlayerId = new Map(players.map((player) => [player.playerId, player.token]));

  return (
    <section className="board-grid-surface">
      <div className="board-grid__header">
        <p className="eyebrow">Board</p>
        <p className="board-grid__actor-label">{actorLabel}</p>
      </div>
      <h2 className="board-grid__title">{gameLabel} board</h2>
      <div
        role="grid"
        aria-label={`${gameLabel} board`}
        className="board-grid"
        style={{ gridTemplateColumns: `repeat(${boardSize}, minmax(0, 1fr))` }}
      >
        {orderedCells.map((cell) => {
          const token = resolveCellToken(cell, tokenByPlayerId);
          const isOccupied = token !== "" || cell.playerId !== null;
          const isLegalEmpty =
            !isOccupied && (cell.isLegalAction || legalCoords.has(cell.coord));
          const isEnabled = canSubmitMoves && isLegalEmpty;
          const className = [
            "board-grid__cell",
            isOccupied ? "board-grid__cell--occupied" : "",
            isLegalEmpty ? "board-grid__cell--legal" : "",
            cell.isLastMove ? "board-grid__cell--last-move" : "",
            cell.isWinningCell ? "board-grid__cell--winning" : "",
          ]
            .filter((name) => name !== "")
            .join(" ");

          return (
            <button
              key={cell.coord}
              type="button"
              data-testid={`board-cell-${cell.coord}`}
              className={className}
              data-last-move={cell.isLastMove ? "true" : "false"}
              data-winning-cell={cell.isWinningCell ? "true" : "false"}
              disabled={!isEnabled}
              aria-label={`Board cell ${cell.coord}`}
              onClick={() => {
                onSubmitMove(cell.coord);
              }}
            >
              <span className="board-grid__token">{token || "·"}</span>
              <span className="board-grid__coord">{cell.coord}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
