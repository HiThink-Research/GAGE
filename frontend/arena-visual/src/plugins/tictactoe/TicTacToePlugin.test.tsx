import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { VisualScene } from "../../gateway/types";
import tictactoeScene from "../../test/fixtures/tictactoe.visual.json";
import { TicTacToePlugin } from "./TicTacToePlugin";

describe("TicTacToePlugin", () => {
  it("renders non-blank board state, winning cells, and blocks coord actions when input is closed", () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);

    render(
      <TicTacToePlugin
        session={{
          sessionId: "ttt-sample",
          gameId: "tictactoe",
          pluginId: "arena.visualization.tictactoe.board_v1",
          lifecycle: "closed",
          playback: {
            mode: "paused",
            cursorTs: 1003,
            cursorEventSeq: 3,
            speed: 1,
            canSeek: true
          },
          observer: {
            observerId: "spectator-cam",
            observerKind: "spectator"
          },
          scheduling: {
            family: "turn",
            phase: "completed",
            acceptsHumanIntent: false,
            activeActorId: "player_1"
          },
          capabilities: {},
          summary: {},
          timeline: {}
        }}
        scene={tictactoeScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={submitInput}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByRole("heading", { name: /tic-tac-toe board/i })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /board cell/i })).toHaveLength(9);
    expect(screen.getByRole("grid", { name: /tic-tac-toe board/i })).toHaveTextContent("X");
    expect(screen.getByRole("grid", { name: /tic-tac-toe board/i })).toHaveTextContent("O");
    expect(screen.getByText(/active player: player_1/i)).toBeInTheDocument();

    const winningCell = screen.getByRole("button", { name: /board cell 3,3/i });
    expect(winningCell.getAttribute("data-winning-cell")).toBe("true");

    const actionButton = screen.getByRole("button", { name: /board cell 1,2/i });
    expect(actionButton).toBeDisabled();
    fireEvent.click(actionButton);
    expect(submitInput).not.toHaveBeenCalled();
  });
});
