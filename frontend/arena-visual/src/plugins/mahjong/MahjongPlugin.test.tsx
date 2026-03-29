import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { VisualScene } from "../../gateway/types";
import mahjongScene from "../../test/fixtures/mahjong.visual.json";
import { MahjongPlugin } from "./MahjongPlugin";

describe("MahjongPlugin", () => {
  it("renders a non-blank discard table, masks spectator hands, and blocks tile actions when input is closed", () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);

    render(
      <MahjongPlugin
        session={{
          sessionId: "mahjong-sample",
          gameId: "mahjong",
          pluginId: "arena.visualization.mahjong.table_v1",
          lifecycle: "closed",
          playback: {
            mode: "paused",
            cursorTs: 1009,
            cursorEventSeq: 9,
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
            activeActorId: "east",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={mahjongScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={submitInput}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByRole("heading", { name: /mahjong table/i })).toBeInTheDocument();
    expect(screen.getByTestId("table-center-cards")).toHaveTextContent("B1");
    expect(screen.getByTestId("table-center-cards")).toHaveTextContent("C1");
    expect(screen.getByTestId("seat-east-hand")).toHaveTextContent("Hidden hand");
    expect(screen.getByText(/active player: east/i)).toBeInTheDocument();
    expect(screen.getByTestId("seat-south-notes")).toHaveTextContent("Pong C3");

    const playButton = screen.getByRole("button", { name: /play b1/i });
    expect(playButton).toBeDisabled();
    fireEvent.click(playButton);
    expect(submitInput).not.toHaveBeenCalled();
  });

  it("renders a rich table with compass seats, tile art, and meld panels", () => {
    render(
      <MahjongPlugin
        session={{
          sessionId: "mahjong-sample",
          gameId: "mahjong",
          pluginId: "arena.visualization.mahjong.table_v1",
          lifecycle: "closed",
          playback: {
            mode: "paused",
            cursorTs: 1009,
            cursorEventSeq: 9,
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
            activeActorId: "east",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={mahjongScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={vi.fn()}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByLabelText("Table seat east")).toBeInTheDocument();
    expect(screen.getByLabelText("Table seat south")).toBeInTheDocument();
    expect(screen.getByText("Discards")).toBeInTheDocument();
    expect(screen.getAllByAltText("B1").length).toBeGreaterThan(0);
    expect(screen.getByText("Pong C3")).toBeInTheDocument();
  });
});
