import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { fireEvent, render, screen, within } from "@testing-library/react";
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
    expect(screen.getByTestId("mahjong-stage-status")).toHaveTextContent("Wall 61");
    expect(screen.getByTestId("mahjong-discard-pool")).toHaveTextContent("B1");
    expect(screen.getByTestId("mahjong-discard-pool")).toHaveTextContent("C1");
    expect(screen.getByTestId("mahjong-history")).toHaveTextContent("South melded Pong C3");
    expect(screen.getByTestId("mahjong-seat-right-bubble")).toHaveTextContent("pon");
    expect(screen.getByTestId("mahjong-seat-bottom-hand")).toHaveTextContent("Hidden hand");
    expect(screen.getByText("Pong C3")).toBeInTheDocument();
    expect(screen.getByLabelText("Mahjong seat east")).toBeInTheDocument();

    expect(screen.queryByRole("button", { name: /select b1/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /confirm b1/i })).toBeNull();
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
      screen.getByTestId("mahjong-seat-bottom-hand").querySelector('[aria-label="Select B1"]'),
    ).not.toBeNull();
    expect(
      screen.getByTestId("mahjong-draw-slot").querySelector('[aria-label="Select Red"]'),
    ).not.toBeNull();
    expect(screen.getByText("Chow B2-B3-B4")).toBeInTheDocument();
    expect(screen.getByText("Kong D5")).toBeInTheDocument();
    expect(screen.getByTestId("mahjong-call-panel")).toBeInTheDocument();
    expect(screen.getByTestId("mahjong-history")).toHaveTextContent("East · B1");
    expect(screen.getByTestId("mahjong-seat-right-bubble")).toHaveTextContent("pon");
    expect(screen.getByTestId("mahjong-stage-status")).toHaveTextContent("Viewing East");
    expect(screen.getByTestId("mahjong-stage-status")).toHaveTextContent("Wall 52");
    expect(within(screen.getByTestId("mahjong-call-panel")).getByRole("button", { name: /play pong/i })).toBeInTheDocument();
    expect(within(screen.getByTestId("mahjong-call-panel")).getByRole("button", { name: /play hu/i })).toBeInTheDocument();
    expect(
      screen
        .getByTestId("mahjong-seat-bottom-hand")
        .querySelectorAll(".mahjong-hand__rack > .mahjong-tile"),
    ).toHaveLength(13);
    expect(screen.getByTestId("mahjong-seat-bottom-hand")).toHaveTextContent("B9");
    expect(screen.getByTestId("mahjong-seat-bottom-hand")).toHaveTextContent("Green");

    fireEvent.doubleClick(screen.getByRole("button", { name: /select b1/i }));
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

  it("renders structured discard lanes and requires confirming a selected tile before submit", () => {
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

    expect(screen.getByTestId("mahjong-discard-lane-east")).toHaveTextContent("B1");
    expect(screen.getByTestId("mahjong-discard-lane-north")).toHaveTextContent("White");
    expect(screen.getByTestId("mahjong-discard-lane-north")).toHaveTextContent("Tedashi");
    expect(screen.getByLabelText(/tedashi discard/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /select b1/i }));
    expect(submitInput).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /confirm b1/i })).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: /confirm b1/i }));
    expect(submitInput).toHaveBeenCalledWith({
      playerId: "east",
      actionText: "B1",
    });
  });

  it("renders a result banner when the scene carries terminal hand semantics", () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);
    const resultScene = structuredClone(mahjongRichScene) as VisualScene;
    const resultBody = resultScene.body as Record<string, any>;

    resultBody.status = {
      ...(resultBody.status as Record<string, any>),
      winner: "east",
      result: "win",
      resultReason: "self_draw",
      remainingTiles: 37,
    };

    render(
      <MahjongPlugin
        session={{
          sessionId: "mahjong-result",
          gameId: "mahjong",
          pluginId: "arena.visualization.mahjong.table_v1",
          lifecycle: "closed",
          playback: {
            mode: "paused",
            cursorTs: 3021,
            cursorEventSeq: 30,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "east",
            observerKind: "player",
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
        scene={resultScene}
        submitAction={vi.fn()}
        submitInput={submitInput}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByTestId("mahjong-result-banner")).toHaveTextContent("East Win");
    expect(screen.getByTestId("mahjong-result-banner")).toHaveTextContent("Self Draw");
    expect(screen.getByTestId("mahjong-result-banner")).toHaveTextContent("37 tiles left in wall");
    expect(screen.getByTestId("mahjong-stage-status")).toHaveTextContent("East Win");
    expect(submitInput).not.toHaveBeenCalled();
  });
});
