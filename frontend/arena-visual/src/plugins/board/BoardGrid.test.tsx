import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { BoardCell, BoardPlayer } from "./boardScene";
import { BoardGrid } from "./BoardGrid";

const PLAYERS: BoardPlayer[] = [
  { playerId: "player-1", playerName: "Player 1", token: "X" },
  { playerId: "player-2", playerName: "Player 2", token: "O" },
];

function buildCell(
  coord: string,
  row: number,
  col: number,
  overrides: Partial<BoardCell> = {},
): BoardCell {
  return {
    coord,
    row,
    col,
    occupant: null,
    playerId: null,
    playerName: null,
    isLastMove: false,
    isWinningCell: false,
    isLegalAction: false,
    ...overrides,
  };
}

describe("BoardGrid", () => {
  it("preserves top-to-bottom row order for A1 boards", () => {
    const cells = [
      buildCell("A3", 2, 0),
      buildCell("B3", 2, 1),
      buildCell("C3", 2, 2),
      buildCell("A2", 1, 0),
      buildCell("B2", 1, 1, { occupant: "O", playerId: "player-2", isLastMove: true }),
      buildCell("C2", 1, 2),
      buildCell("A1", 0, 0, { occupant: "X", playerId: "player-1" }),
      buildCell("B1", 0, 1),
      buildCell("C1", 0, 2),
    ];

    render(
      <BoardGrid
        variant="gomoku"
        gameLabel="Gomoku"
        actorLabel="Observer: player-1"
        boardSize={3}
        coordScheme="A1"
        cells={cells}
        players={PLAYERS}
        status={{
          activePlayerId: "player-2",
          observerPlayerId: "player-1",
          moveCount: 2,
          lastMove: "B2",
          winningLine: ["A1", "B2"],
        }}
        legalCoords={new Set<string>(["B1"])}
        canSubmitMoves={true}
        onSubmitMove={vi.fn()}
      />,
    );

    const buttons = screen.getAllByRole("button", { name: /board cell/i });
    expect(buttons.slice(0, 3).map((button) => button.getAttribute("aria-label"))).toEqual([
      "Board cell A3",
      "Board cell B3",
      "Board cell C3",
    ]);
    expect(buttons.slice(-3).map((button) => button.getAttribute("aria-label"))).toEqual([
      "Board cell A1",
      "Board cell B1",
      "Board cell C1",
    ]);
    expect(screen.getByRole("button", { name: /board cell b2/i }).getAttribute("data-last-move")).toBe("true");
  });

  it("preserves top-to-bottom row order for ROW_COL boards", () => {
    const cells = [
      buildCell("3,1", 2, 0),
      buildCell("3,2", 2, 1),
      buildCell("3,3", 2, 2, { isWinningCell: true }),
      buildCell("2,1", 1, 0),
      buildCell("2,2", 1, 1, { occupant: "O", playerId: "player-2", isLastMove: true }),
      buildCell("2,3", 1, 2),
      buildCell("1,1", 0, 0, { occupant: "X", playerId: "player-1" }),
      buildCell("1,2", 0, 1),
      buildCell("1,3", 0, 2),
    ];

    render(
      <BoardGrid
        variant="tictactoe"
        gameLabel="Tic-Tac-Toe"
        actorLabel="Active player: player-2"
        boardSize={3}
        coordScheme="ROW_COL"
        cells={cells}
        players={PLAYERS}
        status={{
          activePlayerId: "player-2",
          observerPlayerId: null,
          moveCount: 2,
          lastMove: "2,2",
          winningLine: ["1,1", "2,2", "3,3"],
        }}
        legalCoords={new Set<string>(["1,2"])}
        canSubmitMoves={false}
        onSubmitMove={vi.fn()}
      />,
    );

    const buttons = screen.getAllByRole("button", { name: /board cell/i });
    expect(buttons.slice(0, 3).map((button) => button.getAttribute("aria-label"))).toEqual([
      "Board cell 3,1",
      "Board cell 3,2",
      "Board cell 3,3",
    ]);
    expect(buttons.slice(-3).map((button) => button.getAttribute("aria-label"))).toEqual([
      "Board cell 1,1",
      "Board cell 1,2",
      "Board cell 1,3",
    ]);
    expect(screen.getByRole("button", { name: /board cell 2,2/i }).getAttribute("data-last-move")).toBe("true");
  });
});
