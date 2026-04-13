import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { VisualScene } from "../../gateway/types";
import gomokuScene from "../../test/fixtures/gomoku.visual.json";
import gomokuRichScene from "../../test/fixtures/gomoku.rich.visual";
import { readBoardScene, readLegalCoords } from "../board/boardScene";
import { GomokuBoard } from "./GomokuBoard";

describe("GomokuBoard", () => {
  const richBoardScene = readBoardScene(gomokuRichScene as VisualScene);
  if (!richBoardScene) {
    throw new Error("rich gomoku fixture did not parse into a board scene");
  }

  it("renders a dense stage-only gomoku board with four-sided coordinates, star points, and a winning-line overlay", () => {
    const container = render(
      <GomokuBoard
        board={richBoardScene.board}
        legalCoords={readLegalCoords(gomokuRichScene as VisualScene)}
        winningLine={richBoardScene.status.winningLine}
        canSubmitMoves={false}
        onSubmitMove={() => {}}
      />,
    ).container;

    expect(screen.getByTestId("gomoku-stage")).toBeInTheDocument();
    expect(container.querySelector(".gomoku-board-frame")).not.toBeNull();
    expect(container.querySelectorAll(".gomoku-board__coords--top .gomoku-board__coord")).toHaveLength(15);
    expect(container.querySelectorAll(".gomoku-board__coords--bottom .gomoku-board__coord")).toHaveLength(15);
    expect(container.querySelectorAll(".gomoku-board__coords--left .gomoku-board__coord")).toHaveLength(15);
    expect(container.querySelectorAll(".gomoku-board__coords--right .gomoku-board__coord")).toHaveLength(15);
    expect(screen.getAllByRole("button", { name: /board cell/i })).toHaveLength(225);

    const firstVisualCell = screen.getAllByRole("button", { name: /board cell/i })[0];
    expect(firstVisualCell).toHaveAttribute("aria-label", "Board cell A15");

    expect(screen.getByTestId("gomoku-winning-line")).toBeInTheDocument();
    expect(container.querySelectorAll(".gomoku-board__star-point")).toHaveLength(9);

    const lastMoveCell = screen.getByTestId("board-cell-I10");
    expect(lastMoveCell).toHaveAttribute("data-last-move", "true");
    expect(within(lastMoveCell).getByText("B")).toHaveClass("gomoku-board__stone--black");
    expect(lastMoveCell.querySelector(".gomoku-board__last-move")).not.toBeNull();

    const whiteCell = screen.getByTestId("board-cell-H9");
    expect(within(whiteCell).getByText("W")).toHaveClass("gomoku-board__stone--white");

    const starPointCell = screen.getByTestId("board-cell-D4");
    expect(starPointCell.querySelector(".gomoku-board__star-point")).not.toBeNull();
  });

  it("submits only legal empty intersections when the board is interactive", () => {
    const onSubmitMove = vi.fn();
    const scene = gomokuScene as VisualScene;
    const boardScene = readBoardScene(scene);

    if (!boardScene) {
      throw new Error("fixture did not parse into a board scene");
    }

    render(
      <GomokuBoard
        board={boardScene.board}
        legalCoords={readLegalCoords(scene)}
        winningLine={boardScene.status.winningLine}
        canSubmitMoves={true}
        onSubmitMove={onSubmitMove}
      />,
    );

    const legalCell = screen.getByTestId("board-cell-B1");
    const occupiedCell = screen.getByTestId("board-cell-A1");

    expect(legalCell).toBeEnabled();
    expect(occupiedCell).toBeDisabled();

    fireEvent.click(legalCell);
    fireEvent.click(occupiedCell);

    expect(onSubmitMove).toHaveBeenCalledTimes(1);
    expect(onSubmitMove).toHaveBeenCalledWith("B1");
  });
});
