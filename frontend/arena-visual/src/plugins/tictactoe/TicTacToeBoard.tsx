import type { BoardCell, BoardSceneData } from "../board/boardScene";

import "./tictactoe.css";

interface TicTacToeBoardProps {
  board: BoardSceneData["board"];
  legalCoords: Set<string>;
  canSubmitMoves: boolean;
  onSubmitMove: (coord: string) => void;
}

interface CoordParts {
  column: string;
  row: string;
}

function parseCoordParts(coord: string, coordScheme: string): CoordParts {
  if ((coordScheme === "ROW_COL" || coord.includes(",")) && coord.includes(",")) {
    const [row, column] = coord.split(",", 2).map((part) => part.trim());
    return {
      column: column || coord,
      row: row || coord,
    };
  }

  const match = coord.trim().match(/^([A-Za-z]+)(\d+)$/);
  if (match) {
    return {
      column: match[1].toUpperCase(),
      row: match[2],
    };
  }

  return {
    column: coord,
    row: coord,
  };
}

function buildCoordRails(cells: BoardCell[], boardSize: number, coordScheme: string) {
  const columns = Array.from({ length: boardSize }, (_, index) => {
    const sample = cells.find((cell) => cell.col === index);
    if (sample) {
      return parseCoordParts(sample.coord, coordScheme).column;
    }
    return coordScheme === "ROW_COL" ? String(index + 1) : String.fromCharCode(65 + index);
  });

  const rows = Array.from({ length: boardSize }, (_, index) => {
    const rowIndex = boardSize - index - 1;
    const sample = cells.find((cell) => cell.row === rowIndex);
    if (sample) {
      return parseCoordParts(sample.coord, coordScheme).row;
    }
    return String(rowIndex + 1);
  });

  return { columns, rows };
}

function orderCellsForDisplay(cells: BoardCell[]): BoardCell[] {
  return [...cells].sort((left, right) => {
    if (left.row !== right.row) {
      return right.row - left.row;
    }
    return left.col - right.col;
  });
}

function resolveTokenClass(token: string | null): string {
  if (token === "X") {
    return "tictactoe-board-cell__token--x";
  }
  if (token === "O") {
    return "tictactoe-board-cell__token--o";
  }
  return "tictactoe-board-cell__token--empty";
}

export function TicTacToeBoard({
  board,
  legalCoords,
  canSubmitMoves,
  onSubmitMove,
}: TicTacToeBoardProps) {
  const rails = buildCoordRails(board.cells, board.size, board.coordScheme);
  const displayCells = orderCellsForDisplay(board.cells);

  return (
    <section className="tictactoe-stage" data-testid="tictactoe-stage">
      <div className="tictactoe-stage__board-wrap">
        <div className="tictactoe-board-frame">
          <div className="tictactoe-board__coords tictactoe-board__coords--top" aria-hidden="true">
            {rails.columns.map((label) => (
              <span key={`top-${label}`} className="tictactoe-board__coord">
                {label}
              </span>
            ))}
          </div>

          <div className="tictactoe-board__main">
            <div className="tictactoe-board__coords tictactoe-board__coords--left" aria-hidden="true">
              {rails.rows.map((label) => (
                <span key={`left-${label}`} className="tictactoe-board__coord">
                  {label}
                </span>
              ))}
            </div>

            <div
              role="grid"
              aria-label="Tic-Tac-Toe board"
              className="tictactoe-board-grid"
              style={{ gridTemplateColumns: `repeat(${board.size}, minmax(0, 1fr))` }}
            >
              {displayCells.map((cell) => {
                const token = cell.occupant;
                const isOccupied = token !== null && token.trim() !== "";
                const isLegalEmpty = !isOccupied && (cell.isLegalAction || legalCoords.has(cell.coord));
                const isEnabled = canSubmitMoves && isLegalEmpty;
                const className = [
                  "tictactoe-board-cell",
                  isOccupied ? "is-occupied" : "",
                  isLegalEmpty ? "is-legal" : "",
                  cell.isLastMove ? "is-last-move" : "",
                  cell.isWinningCell ? "is-winning" : "",
                ]
                  .filter((value) => value !== "")
                  .join(" ");

                return (
                  <button
                    key={cell.coord}
                    type="button"
                    className={className}
                    data-testid={`tictactoe-cell-${cell.coord}`}
                    data-last-move={cell.isLastMove ? "true" : "false"}
                    data-winning-cell={cell.isWinningCell ? "true" : "false"}
                    aria-label={`Board cell ${cell.coord}`}
                    disabled={!isEnabled}
                    onClick={() => {
                      onSubmitMove(cell.coord);
                    }}
                  >
                    {isOccupied ? (
                      <span className={`tictactoe-board-cell__token ${resolveTokenClass(token)}`}>{token}</span>
                    ) : null}
                    {!isOccupied && canSubmitMoves && isLegalEmpty ? (
                      <span className="tictactoe-board-cell__hint" aria-hidden="true">
                        •
                      </span>
                    ) : null}
                  </button>
                );
              })}
            </div>

            <div className="tictactoe-board__coords tictactoe-board__coords--right" aria-hidden="true">
              {rails.rows.map((label) => (
                <span key={`right-${label}`} className="tictactoe-board__coord">
                  {label}
                </span>
              ))}
            </div>
          </div>

          <div className="tictactoe-board__coords tictactoe-board__coords--bottom" aria-hidden="true">
            {rails.columns.map((label) => (
              <span key={`bottom-${label}`} className="tictactoe-board__coord">
                {label}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
