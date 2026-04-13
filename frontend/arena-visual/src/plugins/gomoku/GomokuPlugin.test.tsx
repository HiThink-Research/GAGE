import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { VisualScene } from "../../gateway/types";
import gomokuScene from "../../test/fixtures/gomoku.visual.json";
import { GomokuPlugin } from "./GomokuPlugin";

describe("GomokuPlugin", () => {
  it("renders only the dedicated gomoku board stage inside the plugin and drops legacy board chrome", () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);

    render(
      <GomokuPlugin
        session={{
          sessionId: "gomoku-sample",
          gameId: "gomoku",
          pluginId: "arena.visualization.gomoku.board_v1",
          lifecycle: "closed",
          playback: {
            mode: "paused",
            cursorTs: 1005,
            cursorEventSeq: 5,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "",
            observerKind: "spectator",
          },
          scheduling: {
            family: "turn",
            phase: "completed",
            acceptsHumanIntent: false,
            activeActorId: "White",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={gomokuScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={submitInput}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByTestId("gomoku-stage")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /gomoku board/i })).not.toBeInTheDocument();
    expect(screen.queryByText("Black stone")).not.toBeInTheDocument();
    expect(screen.queryByText("White stone")).not.toBeInTheDocument();
    expect(screen.queryByText("Winning line")).not.toBeInTheDocument();

    expect(screen.getAllByRole("button", { name: /board cell/i })).toHaveLength(9);

    const actionButton = screen.getByRole("button", { name: /board cell b1/i });
    expect(actionButton).toBeDisabled();
    fireEvent.click(actionButton);
    expect(submitInput).not.toHaveBeenCalled();
  });

  it("submits a player coord from the dedicated stage when human input is enabled", () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);

    render(
      <GomokuPlugin
        session={{
          sessionId: "gomoku-sample",
          gameId: "gomoku",
          pluginId: "arena.visualization.gomoku.board_v1",
          lifecycle: "live_running",
          playback: {
            mode: "paused",
            cursorTs: 1005,
            cursorEventSeq: 5,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "Black",
            observerKind: "player",
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "White",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={gomokuScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={submitInput}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    const actionButton = screen.getByRole("button", { name: /board cell b1/i });
    expect(actionButton).toBeEnabled();

    fireEvent.click(actionButton);

    expect(submitInput).toHaveBeenCalledWith({
      playerId: "Black",
      coord: "B1",
    });
  });
});
