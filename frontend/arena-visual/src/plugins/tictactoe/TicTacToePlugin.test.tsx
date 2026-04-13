import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { VisualScene } from "../../gateway/types";
import tictactoeScene from "../../test/fixtures/tictactoe.visual.json";
import { TicTacToePlugin } from "./TicTacToePlugin";

describe("TicTacToePlugin", () => {
  it("renders only the dedicated board stage inside the plugin and drops legacy board chrome", () => {
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

    expect(screen.getByTestId("tictactoe-stage")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /tic-tac-toe board/i })).not.toBeInTheDocument();
    expect(screen.queryByText("X mark")).not.toBeInTheDocument();
    expect(screen.queryByText("O mark")).not.toBeInTheDocument();
    expect(screen.queryByText("Winning line")).not.toBeInTheDocument();

    expect(screen.getAllByRole("button", { name: /board cell/i })).toHaveLength(9);

    const actionButton = screen.getByRole("button", { name: /board cell 1,2/i });
    expect(actionButton).toBeDisabled();
    fireEvent.click(actionButton);
    expect(submitInput).not.toHaveBeenCalled();
  });

  it("submits a player coord from the dedicated stage when human input is enabled", () => {
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
            observerId: "player_0",
            observerKind: "player"
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
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

    const actionButton = screen.getByRole("button", { name: /board cell 1,2/i });
    expect(actionButton).toBeEnabled();

    fireEvent.click(actionButton);

    expect(submitInput).toHaveBeenCalledWith({
      playerId: "player_0",
      coord: "1,2",
    });
  });
});
