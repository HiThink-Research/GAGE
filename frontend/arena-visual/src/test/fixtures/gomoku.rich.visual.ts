import type { VisualScene } from "../../gateway/types";

const boardSize = 15;
const winningLine = ["E6", "F7", "G8", "H9", "I10"];
const legalCoords = new Set(["D6", "F6", "G6", "J10"]);
const occupiedByCoord = new Map<string, { occupant: "B" | "W"; playerId: string; playerName: string }>([
  ["C3", { occupant: "B", playerId: "Black", playerName: "Black" }],
  ["D5", { occupant: "W", playerId: "White", playerName: "White" }],
  ["E6", { occupant: "B", playerId: "Black", playerName: "Black" }],
  ["F7", { occupant: "W", playerId: "White", playerName: "White" }],
  ["G8", { occupant: "B", playerId: "Black", playerName: "Black" }],
  ["H9", { occupant: "W", playerId: "White", playerName: "White" }],
  ["I10", { occupant: "B", playerId: "Black", playerName: "Black" }],
  ["J4", { occupant: "W", playerId: "White", playerName: "White" }],
  ["K5", { occupant: "B", playerId: "Black", playerName: "Black" }],
  ["L6", { occupant: "W", playerId: "White", playerName: "White" }],
  ["M3", { occupant: "B", playerId: "Black", playerName: "Black" }],
]);

function coordFor(row: number, col: number): string {
  return `${String.fromCharCode(65 + col)}${row + 1}`;
}

const cells = Array.from({ length: boardSize * boardSize }, (_, index) => {
  const row = Math.floor(index / boardSize);
  const col = index % boardSize;
  const coord = coordFor(row, col);
  const occupied = occupiedByCoord.get(coord);

  return {
    coord,
    row,
    col,
    occupant: occupied?.occupant ?? null,
    playerId: occupied?.playerId ?? null,
    playerName: occupied?.playerName ?? null,
    isLastMove: coord === "I10",
    isWinningCell: winningLine.includes(coord),
    isLegalAction: legalCoords.has(coord),
  };
});

const richScene: VisualScene = {
  sceneId: "gomoku:seq:42",
  gameId: "gomoku",
  pluginId: "arena.visualization.gomoku.board_v1",
  kind: "board",
  tsMs: 2042,
  seq: 42,
  phase: "replay",
  activePlayerId: "White",
  legalActions: Array.from(legalCoords, (coord) => {
    const cell = cells.find((candidate) => candidate.coord === coord);
    if (!cell) {
      throw new Error(`Missing cell ${coord}`);
    }
    return {
      id: coord,
      label: coord,
      coord,
      row: cell.row,
      col: cell.col,
    };
  }),
  summary: {
    boardSize,
    coordScheme: "A1",
  },
  body: {
    board: {
      size: boardSize,
      coordScheme: "A1",
      cells,
    },
    players: [
      { playerId: "Black", playerName: "Black", token: "B" },
      { playerId: "White", playerName: "White", token: "W" },
    ],
    status: {
      activePlayerId: "White",
      observerPlayerId: "Black",
      moveCount: occupiedByCoord.size,
      lastMove: "I10",
      winningLine,
    },
  },
};

export default richScene;
