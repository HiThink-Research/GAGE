import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { VisualScene } from "../../gateway/types";
import mahjongScene from "../../test/fixtures/mahjong.visual.json";
import mahjongRichScene from "../../test/fixtures/mahjong.rich.visual";
import { MahjongPlugin } from "./MahjongPlugin";

describe("MahjongPlugin", () => {
  it("uses a grid-based table layout so full hands do not overlap the discard pool", () => {
    const css = readFileSync(resolve(process.cwd(), "src/plugins/mahjong/mahjong.css"), "utf-8");

    expect(css).toMatch(
      /grid-template-areas:\s*"top top top"\s*"left center right"\s*"bottom bottom bottom"/,
    );
    expect(css).toMatch(/\.mahjong-discards\s*\{[^}]*grid-area:\s*center/s);
    expect(css).toMatch(/\.mahjong-seat--bottom\s*\{[^}]*grid-area:\s*bottom/s);
    expect(css).not.toMatch(/\.mahjong-seat\s*\{[^}]*position:\s*absolute/s);
  });

  it("renders a discard pool, parses meld notes, and keeps spectator hands hidden", () => {
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

    expect(screen.getByTestId("mahjong-stage")).toBeInTheDocument();
    expect(screen.getByTestId("mahjong-discard-pool")).toHaveTextContent("B1");
    expect(screen.getByTestId("mahjong-discard-pool")).toHaveTextContent("C1");
    expect(screen.getByTestId("mahjong-seat-bottom-hand")).toHaveTextContent("Hidden hand");
    expect(screen.getByText("Pong C3")).toBeInTheDocument();
    expect(screen.getByLabelText("Mahjong seat east")).toBeInTheDocument();

    const playButton = screen.getByRole("button", { name: /play b1/i });
    expect(playButton).toBeDisabled();
    fireEvent.click(playButton);
    expect(submitInput).not.toHaveBeenCalled();
  });

  it("anchors the observer hand at the bottom and submits tile or action intents from the plugin", () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);

    render(
      <MahjongPlugin
        session={{
          sessionId: "mahjong-sample",
          gameId: "mahjong",
          pluginId: "arena.visualization.mahjong.table_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
            cursorTs: 2021,
            cursorEventSeq: 21,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "east",
            observerKind: "player",
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "east",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={mahjongRichScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={submitInput}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByLabelText("Mahjong seat east")).toHaveAttribute(
      "data-testid",
      "mahjong-seat-bottom",
    );
    expect(
      screen.getByTestId("mahjong-seat-bottom-hand").querySelector('[aria-label="Play B1"]'),
    ).not.toBeNull();
    expect(
      screen.getByTestId("mahjong-draw-slot").querySelector('[aria-label="Play Red"]'),
    ).not.toBeNull();
    expect(screen.getByText("Chow B2-B3-B4")).toBeInTheDocument();
    expect(screen.getByText("Kong D5")).toBeInTheDocument();
    expect(
      screen
        .getByTestId("mahjong-seat-bottom-hand")
        .querySelectorAll(".mahjong-hand__rack > .mahjong-tile"),
    ).toHaveLength(13);
    expect(screen.getByTestId("mahjong-seat-bottom-hand")).toHaveTextContent("B9");
    expect(screen.getByTestId("mahjong-seat-bottom-hand")).toHaveTextContent("Green");

    fireEvent.click(screen.getByRole("button", { name: /play b1/i }));
    expect(submitInput).toHaveBeenCalledWith({
      playerId: "east",
      actionText: "B1",
    });

    fireEvent.click(screen.getByRole("button", { name: /play pong/i }));
    expect(submitInput).toHaveBeenCalledWith({
      playerId: "east",
      actionText: "Pong",
    });
  });
});
