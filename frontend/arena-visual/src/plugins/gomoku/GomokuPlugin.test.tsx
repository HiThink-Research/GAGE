import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { VisualScene } from "../../gateway/types";
import gomokuScene from "../../test/fixtures/gomoku.visual.json";
import { GomokuPlugin } from "./GomokuPlugin";

describe("GomokuPlugin", () => {
  it("renders a visible board, highlights the last move, and submits coord intents", () => {
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
            canSeek: true
          },
          observer: {
            observerId: "Black",
            observerKind: "player"
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "White"
          },
          capabilities: {},
          summary: {},
          timeline: {}
        }}
        scene={gomokuScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={submitInput}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByRole("heading", { name: /gomoku board/i })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /board cell/i })).toHaveLength(9);
    expect(screen.getByRole("grid", { name: /gomoku board/i })).toHaveTextContent("B");
    expect(screen.getByRole("grid", { name: /gomoku board/i })).toHaveTextContent("W");
    expect(screen.getByText(/observer: black/i)).toBeInTheDocument();

    const lastMoveCell = screen.getByRole("button", { name: /board cell b2/i });
    expect(lastMoveCell.getAttribute("data-last-move")).toBe("true");

    fireEvent.click(screen.getByRole("button", { name: /board cell b1/i }));
    expect(submitInput).toHaveBeenCalledWith({
      playerId: "Black",
      coord: "B1",
    });
  });

  it("renders rich board chrome with player cards, coordinate rails, and board summary", () => {
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
            canSeek: true
          },
          observer: {
            observerId: "Black",
            observerKind: "player"
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "White"
          },
          capabilities: {},
          summary: {},
          timeline: {}
        }}
        scene={gomokuScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={vi.fn()}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByText("Black stone")).toBeInTheDocument();
    expect(screen.getByText("White stone")).toBeInTheDocument();
    expect(screen.getByText("Winning line")).toBeInTheDocument();
    expect(screen.getByText("A1 -> B2")).toBeInTheDocument();
    expect(screen.getAllByLabelText("Board column A")).toHaveLength(2);
    expect(screen.getAllByLabelText("Board row 3")).toHaveLength(2);
  });
});
