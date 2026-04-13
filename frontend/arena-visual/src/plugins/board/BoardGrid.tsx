import type { CSSProperties } from "react";

import type { BoardCell, BoardPlayer, BoardStatus } from "./boardScene";

export type BoardVariant = "gomoku" | "tictactoe";

interface BoardGridProps {
  variant: BoardVariant;
  gameLabel: string;
  actorLabel: string;
  boardSize: number;
  coordScheme: string;
  cells: BoardCell[];
  players: BoardPlayer[];
  status: BoardStatus;
  legalCoords: Set<string>;
  canSubmitMoves: boolean;
  onSubmitMove: (coord: string) => void;
}

interface CoordParts {
  column: string;
  row: string;
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

function resolvePlayerDescriptor(player: BoardPlayer, variant: BoardVariant): string {
  if (variant === "gomoku") {
    if (player.token === "B") {
      return "Black stone";
    }
    if (player.token === "W") {
      return "White stone";
    }
    return `${player.token} stone`;
  }

  if (player.token === "X") {
    return "X mark";
  }
  if (player.token === "O") {
    return "O mark";
  }
  return `${player.token} mark`;
}

function resolveToneClass(token: string, variant: BoardVariant): string {
  if (variant === "gomoku") {
    if (token === "B") {
      return "black";
    }
    if (token === "W") {
      return "white";
    }
    return "neutral";
  }

  if (token === "X") {
    return "cross";
  }
  if (token === "O") {
    return "nought";
  }
  return "neutral";
}

function formatWinningLine(winningLine: string[]): string {
  if (winningLine.length === 0) {
    return "Pending";
  }
  return winningLine.join(" -> ");
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

function isGomokuStarPoint(cell: BoardCell, boardSize: number): boolean {
  const offsets = resolveStarPointOffsets(boardSize);
  return offsets.includes(cell.row) && offsets.includes(cell.col);
}

export function BoardGrid({
  variant,
  gameLabel,
  actorLabel,
  boardSize,
  coordScheme,
  cells,
  players,
  status,
  legalCoords,
  canSubmitMoves,
  onSubmitMove,
}: BoardGridProps) {
  const tokenByPlayerId = new Map(players.map((player) => [player.playerId, player.token]));
  const rails = buildCoordRails(cells, boardSize, coordScheme);
  const legalMoveCount = cells.filter((cell) => !cell.occupant && (cell.isLegalAction || legalCoords.has(cell.coord))).length;
  const arenaStyle = { "--board-size": String(boardSize) } as CSSProperties;

  return (
    <section className={`board-grid-surface board-grid-surface--${variant}`}>
      <div className="board-grid__header">
        <div>
          <p className="eyebrow">Board</p>
          <h2 className="board-grid__title">{gameLabel} board</h2>
        </div>
        <p className="board-grid__actor-label">{actorLabel}</p>
      </div>

      <div className="board-grid__players" aria-label={`${gameLabel} players`}>
        {players.map((player) => {
          const descriptor = resolvePlayerDescriptor(player, variant);
          const toneClass = resolveToneClass(player.token, variant);
          const cardClassName = [
            "board-grid__player-card",
            `board-grid__player-card--${variant}`,
            `board-grid__player-card--${toneClass}`,
            status.activePlayerId === player.playerId ? "is-active" : "",
          ]
            .filter((name) => name !== "")
            .join(" ");

          return (
            <article
              key={player.playerId}
              className={cardClassName}
              aria-label={`Player card ${player.playerName || player.playerId}`}
            >
              <p className="board-grid__player-descriptor">{descriptor}</p>
              <div className="board-grid__player-main">
                <span className={`board-grid__player-token board-grid__player-token--${variant} board-grid__player-token--${toneClass}`}>
                  {player.token}
                </span>
                <div className="board-grid__player-copy">
                  <strong>{player.playerName || player.playerId}</strong>
                  <span>{player.playerId}</span>
                </div>
              </div>
              <div className="board-grid__player-badges">
                {status.activePlayerId === player.playerId ? (
                  <span className="board-grid__badge board-grid__badge--active">Turn</span>
                ) : null}
                {status.observerPlayerId === player.playerId ? (
                  <span className="board-grid__badge">Observer</span>
                ) : null}
              </div>
            </article>
          );
        })}
      </div>

      <div className="board-grid__layout">
        <div
          className={`board-grid__arena board-grid__arena--${variant}`}
          style={arenaStyle}
        >
          <div className="board-grid__coord-row board-grid__coord-row--top">
            <span className="board-grid__coord-corner" />
            {rails.columns.map((label) => (
              <span
                key={`top-${label}`}
                className="board-grid__coord-rail board-grid__coord-rail--column"
                aria-label={`Board column ${label}`}
              >
                {label}
              </span>
            ))}
            <span className="board-grid__coord-corner" />
          </div>

          <div className="board-grid__board-shell">
            <div className="board-grid__coord-col">
              {rails.rows.map((label) => (
                <span
                  key={`left-${label}`}
                  className="board-grid__coord-rail board-grid__coord-rail--row"
                  aria-label={`Board row ${label}`}
                >
                  {label}
                </span>
              ))}
            </div>

            <div
              role="grid"
              aria-label={`${gameLabel} board`}
              className={`board-grid board-grid--${variant}`}
              style={{ gridTemplateColumns: `repeat(${boardSize}, minmax(0, 1fr))` }}
            >
              {cells.map((cell) => {
                const token = resolveCellToken(cell, tokenByPlayerId);
                const toneClass = resolveToneClass(token, variant);
                const isOccupied = token !== "" || cell.playerId !== null;
                const isLegalEmpty = !isOccupied && (cell.isLegalAction || legalCoords.has(cell.coord));
                const isEnabled = canSubmitMoves && isLegalEmpty;
                const className = [
                  "board-grid__cell",
                  `board-grid__cell--${variant}`,
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
                    <span className="board-grid__cell-surface">
                      {variant === "gomoku" && !isOccupied && isGomokuStarPoint(cell, boardSize) ? (
                        <span className="board-grid__star-point" aria-hidden="true" />
                      ) : null}
                      {isOccupied ? (
                        <span
                          className={`board-grid__piece board-grid__piece--${variant} board-grid__piece--${toneClass}`}
                          aria-hidden="true"
                        >
                          {token}
                        </span>
                      ) : null}
                      {isLegalEmpty ? (
                        <span
                          className={`board-grid__ghost board-grid__ghost--${variant}`}
                          aria-hidden="true"
                        />
                      ) : null}
                    </span>
                    <span className="board-grid__coord">{cell.coord}</span>
                  </button>
                );
              })}
            </div>

            <div className="board-grid__coord-col">
              {rails.rows.map((label) => (
                <span
                  key={`right-${label}`}
                  className="board-grid__coord-rail board-grid__coord-rail--row"
                  aria-label={`Board row ${label}`}
                >
                  {label}
                </span>
              ))}
            </div>
          </div>

          <div className="board-grid__coord-row board-grid__coord-row--bottom">
            <span className="board-grid__coord-corner" />
            {rails.columns.map((label) => (
              <span
                key={`bottom-${label}`}
                className="board-grid__coord-rail board-grid__coord-rail--column"
                aria-label={`Board column ${label}`}
              >
                {label}
              </span>
            ))}
            <span className="board-grid__coord-corner" />
          </div>
        </div>

        <aside className="board-grid__summary" aria-label={`${gameLabel} board summary`}>
          <p className="eyebrow">Snapshot</p>
          <dl className="board-grid__summary-list">
            <div>
              <dt>Move count</dt>
              <dd>{status.moveCount}</dd>
            </div>
            <div>
              <dt>Last move</dt>
              <dd>{status.lastMove ?? "Opening"}</dd>
            </div>
            <div>
              <dt>Winning line</dt>
              <dd>{formatWinningLine(status.winningLine)}</dd>
            </div>
            <div>
              <dt>Playable cells</dt>
              <dd>{legalMoveCount}</dd>
            </div>
          </dl>
        </aside>
      </div>
    </section>
  );
}
