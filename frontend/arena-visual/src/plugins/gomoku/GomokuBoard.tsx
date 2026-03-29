import type { CSSProperties } from "react";

import type { BoardCell, BoardSceneData } from "../board/boardScene";

import "./gomoku.css";

interface GomokuBoardProps {
  board: BoardSceneData["board"];
  legalCoords: Set<string>;
  winningLine: string[];
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

function resolveStarPointOffsets(boardSize: number): number[] {
  if (boardSize >= 15) {
    return [3, Math.floor(boardSize / 2), boardSize - 4];
  }
  if (boardSize >= 9) {
    return [2, Math.floor(boardSize / 2), boardSize - 3];
  }
  if (boardSize >= 5) {
    return [Math.floor(boardSize / 2)];
  }
  return [];
}

function isStarPoint(cell: BoardCell, boardSize: number): boolean {
  const offsets = resolveStarPointOffsets(boardSize);
  return offsets.includes(cell.row) && offsets.includes(cell.col);
}

function resolveStoneClass(token: string | null): string {
  if (token === "B") {
    return "gomoku-board__stone--black";
  }
  if (token === "W") {
    return "gomoku-board__stone--white";
  }
  return "gomoku-board__stone--empty";
}

function resolveWinningLineLayout(
  cells: BoardCell[],
  boardSize: number,
  winningLine: string[],
): CSSProperties | undefined {
  if (winningLine.length < 2) {
    return undefined;
  }

  const cellByCoord = new Map(cells.map((cell) => [cell.coord, cell]));
  const startCell = cellByCoord.get(winningLine[0]);
  const endCell = cellByCoord.get(winningLine[winningLine.length - 1]);
  if (!startCell || !endCell) {
    return undefined;
  }

  const startDisplayRow = boardSize - startCell.row - 1;
  const endDisplayRow = boardSize - endCell.row - 1;
  const startX = ((startCell.col + 0.5) / boardSize) * 100;
  const endX = ((endCell.col + 0.5) / boardSize) * 100;
  const startY = ((startDisplayRow + 0.5) / boardSize) * 100;
  const endY = ((endDisplayRow + 0.5) / boardSize) * 100;
  const dx = endX - startX;
  const dy = endY - startY;
  const length = Math.sqrt(dx * dx + dy * dy);
  const angle = Math.atan2(dy, dx) * (180 / Math.PI);

  return {
    width: `${length}%`,
    left: `${startX}%`,
    top: `${startY}%`,
    transform: `translateY(-50%) rotate(${angle}deg)`,
  };
}

function buildIntersectionStyle(cell: BoardCell, boardSize: number): CSSProperties {
  return {
    gridColumn: cell.col + 1,
    gridRow: boardSize - cell.row,
    "--gomoku-line-x-start": cell.col === 0 ? "50%" : "0%",
    "--gomoku-line-x-end": cell.col === boardSize - 1 ? "50%" : "100%",
    "--gomoku-line-y-start": cell.row === boardSize - 1 ? "50%" : "0%",
    "--gomoku-line-y-end": cell.row === 0 ? "50%" : "100%",
  } as CSSProperties;
}

export function GomokuBoard({
  board,
  legalCoords,
  winningLine,
  canSubmitMoves,
  onSubmitMove,
}: GomokuBoardProps) {
  const rails = buildCoordRails(board.cells, board.size, board.coordScheme);
  const displayCells = orderCellsForDisplay(board.cells);
  const winningLineStyle = resolveWinningLineLayout(board.cells, board.size, winningLine);

  return (
    <section className="gomoku-stage" data-testid="gomoku-stage">
      <div className="gomoku-stage__board-wrap">
        <div className="gomoku-board-frame">
          <div
            className="gomoku-board__coords gomoku-board__coords--top"
            style={{ "--gomoku-board-size": String(board.size) } as CSSProperties}
            aria-hidden="true"
          >
            {rails.columns.map((label) => (
              <span key={`top-${label}`} className="gomoku-board__coord">
                {label}
              </span>
            ))}
          </div>

          <div className="gomoku-board__main">
            <div
              className="gomoku-board__coords gomoku-board__coords--left"
              style={{ "--gomoku-board-size": String(board.size) } as CSSProperties}
              aria-hidden="true"
            >
              {rails.rows.map((label) => (
                <span key={`left-${label}`} className="gomoku-board__coord">
                  {label}
                </span>
              ))}
            </div>

            <div
              role="grid"
              aria-label="Gomoku board"
              className="gomoku-board-surface"
              style={{ "--gomoku-board-size": String(board.size) } as CSSProperties}
            >
              {winningLineStyle ? (
                <div
                  className="gomoku-board__winning-line"
                  data-testid="gomoku-winning-line"
                  style={winningLineStyle}
                  aria-hidden="true"
                />
              ) : null}

              {displayCells.map((cell) => {
                const token = cell.occupant;
                const isOccupied = token !== null && token.trim() !== "";
                const isLegalEmpty = !isOccupied && (cell.isLegalAction || legalCoords.has(cell.coord));
                const isEnabled = canSubmitMoves && isLegalEmpty;
                const className = [
                  "gomoku-board__intersection",
                  isOccupied ? "is-occupied" : "",
                  isLegalEmpty ? "is-legal" : "",
                  cell.isLastMove ? "is-last-move" : "",
                  cell.isWinningCell ? "is-winning" : "",
                  isStarPoint(cell, board.size) ? "is-star-point" : "",
                ]
                  .filter((value) => value !== "")
                  .join(" ");

                return (
                  <button
                    key={cell.coord}
                    type="button"
                    className={className}
                    style={buildIntersectionStyle(cell, board.size)}
                    data-testid={`board-cell-${cell.coord}`}
                    data-last-move={cell.isLastMove ? "true" : "false"}
                    data-winning-cell={cell.isWinningCell ? "true" : "false"}
                    aria-label={`Board cell ${cell.coord}`}
                    disabled={!isEnabled}
                    onClick={() => {
                      onSubmitMove(cell.coord);
                    }}
                  >
                    {!isOccupied && isStarPoint(cell, board.size) ? (
                      <span className="gomoku-board__star-point" aria-hidden="true" />
                    ) : null}
                    {isOccupied ? (
                      <span className={`gomoku-board__stone ${resolveStoneClass(token)}`}>
                        {token}
                        {cell.isLastMove ? <span className="gomoku-board__last-move" aria-hidden="true" /> : null}
                      </span>
                    ) : null}
                  </button>
                );
              })}
            </div>

            <div
              className="gomoku-board__coords gomoku-board__coords--right"
              style={{ "--gomoku-board-size": String(board.size) } as CSSProperties}
              aria-hidden="true"
            >
              {rails.rows.map((label) => (
                <span key={`right-${label}`} className="gomoku-board__coord">
                  {label}
                </span>
              ))}
            </div>
          </div>

          <div
            className="gomoku-board__coords gomoku-board__coords--bottom"
            style={{ "--gomoku-board-size": String(board.size) } as CSSProperties}
            aria-hidden="true"
          >
            {rails.columns.map((label) => (
              <span key={`bottom-${label}`} className="gomoku-board__coord">
                {label}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
