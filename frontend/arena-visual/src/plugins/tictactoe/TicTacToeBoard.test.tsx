import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { VisualScene } from "../../gateway/types";
import tictactoeScene from "../../test/fixtures/tictactoe.visual.json";
import { readBoardScene, readLegalCoords } from "../board/boardScene";
import { TicTacToeBoard } from "./TicTacToeBoard";

describe("TicTacToeBoard", () => {
  const scene = tictactoeScene as VisualScene;
  const boardScene = readBoardScene(scene);

  if (!boardScene) {
    throw new Error("fixture did not parse into a board scene");
  }

  it("renders a stage-only tic-tac-toe board with four-sided coordinates and local styling hooks", () => {
    const container = render(
      <TicTacToeBoard
        board={boardScene.board}
        legalCoords={readLegalCoords(scene)}
        canSubmitMoves={false}
        onSubmitMove={() => {}}
      />,
    ).container;

    expect(screen.getByTestId("tictactoe-stage")).toBeInTheDocument();
    expect(container.querySelector(".tictactoe-board-frame")).not.toBeNull();
    expect(container.querySelectorAll(".tictactoe-board__coords--top .tictactoe-board__coord")).toHaveLength(3);
    expect(container.querySelectorAll(".tictactoe-board__coords--bottom .tictactoe-board__coord")).toHaveLength(3);
    expect(container.querySelectorAll(".tictactoe-board__coords--left .tictactoe-board__coord")).toHaveLength(3);
    expect(container.querySelectorAll(".tictactoe-board__coords--right .tictactoe-board__coord")).toHaveLength(3);

    const firstVisualCell = screen.getAllByRole("button", { name: /board cell/i })[0];
    expect(firstVisualCell).toHaveAttribute("aria-label", "Board cell 3,1");

    const xCell = screen.getByTestId("tictactoe-cell-1,1");
    const oCell = screen.getByTestId("tictactoe-cell-2,2");
    const winningCell = screen.getByTestId("tictactoe-cell-3,3");

    expect(within(xCell).getByText("X")).toHaveClass("tictactoe-board-cell__token--x");
    expect(within(oCell).getByText("O")).toHaveClass("tictactoe-board-cell__token--o");
    expect(oCell).toHaveAttribute("data-last-move", "true");
    expect(winningCell).toHaveAttribute("data-winning-cell", "true");
  });

  it("submits only legal empty coords when the board is interactive", () => {
    const onSubmitMove = vi.fn();

    render(
      <TicTacToeBoard
        board={boardScene.board}
        legalCoords={readLegalCoords(scene)}
        canSubmitMoves={true}
        onSubmitMove={onSubmitMove}
      />,
    );

    const legalCell = screen.getByTestId("tictactoe-cell-1,2");
    const occupiedCell = screen.getByTestId("tictactoe-cell-1,1");

    expect(legalCell).toBeEnabled();
    expect(occupiedCell).toBeDisabled();

    fireEvent.click(legalCell);
    fireEvent.click(occupiedCell);

    expect(onSubmitMove).toHaveBeenCalledTimes(1);
    expect(onSubmitMove).toHaveBeenCalledWith("1,2");
  });
});
