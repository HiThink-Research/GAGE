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
    expect(css).toMatch(/\.mahjong-tile\.is-rotated\s*\{/);
    expect(css).toMatch(/\.mahjong-seat--bottom\s+\.mahjong-seat__bubble\s*\{/);
    expect(css).toMatch(/\.mahjong-seat--top\s+\.mahjong-seat__bubble\s*\{/);
    expect(css).toMatch(/\.mahjong-seat--left\s+\.mahjong-seat__bubble\s*\{/);
    expect(css).toMatch(/\.mahjong-seat--right\s+\.mahjong-seat__bubble\s*\{/);
  });

  it("keeps terminal discard-heavy layouts compact and scroll-safe in fullscreen", () => {
    const css = readFileSync(resolve(process.cwd(), "src/plugins/mahjong/mahjong.css"), "utf-8");

    expect(css).toMatch(/--mahjong-tile-width:\s*clamp\(1\.65rem/s);
    expect(css).toMatch(/\.mahjong-discards\s*\{[^}]*max-height:\s*clamp\(/s);
    expect(css).toMatch(/\.mahjong-discards\s*\{[^}]*overflow-y:\s*auto/s);
    expect(css).toMatch(/\.mahjong-discards__pool\s*\{[^}]*repeat\(auto-fit/s);
    expect(css).toMatch(/\.mahjong-discards\s+\.mahjong-tile\.is-compact\s*\{/s);
    expect(css).toMatch(
      /\.session-stage--fullscreen\s+\.mahjong-stage,\s*\.session-stage:fullscreen\s+\.mahjong-stage\s*\{[^}]*grid-template-rows:\s*auto minmax\(0,\s*1fr\) auto/s,
    );
    expect(css).toMatch(
      /\.session-stage--fullscreen\s+\.mahjong-stage__table,\s*\.session-stage:fullscreen\s+\.mahjong-stage__table\s*\{[^}]*min-height:\s*0/s,
    );
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

  it("renders left and right concealed hands as full vertical stacks", () => {
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
        submitInput={vi.fn()}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(
      screen
        .getByTestId("mahjong-seat-right-hand")
        .querySelectorAll(".mahjong-hand__rack > .mahjong-tile"),
    ).toHaveLength(13);
    expect(
      screen
        .getByTestId("mahjong-seat-left-hand")
        .querySelectorAll(".mahjong-hand__rack > .mahjong-tile"),
    ).toHaveLength(13);
    expect(
      screen
        .getByTestId("mahjong-seat-top-hand")
        .querySelectorAll(".mahjong-hand__rack > .mahjong-tile"),
    ).toHaveLength(13);
    expect(
      screen.getByTestId("mahjong-seat-right-hand").querySelector(".mahjong-hand__rack--vertical"),
    ).not.toBeNull();
    expect(
      screen.getByTestId("mahjong-seat-left-hand").querySelector(".mahjong-hand__rack--vertical"),
    ).not.toBeNull();
    expect(
      screen
        .getByTestId("mahjong-seat-right-hand")
        .querySelectorAll(".mahjong-hand__rack > .mahjong-tile.is-rotated"),
    ).toHaveLength(13);
    expect(
      screen
        .getByTestId("mahjong-seat-left-hand")
        .querySelectorAll(".mahjong-hand__rack > .mahjong-tile.is-rotated"),
    ).toHaveLength(13);
  });

  it("anchors chat bubbles per rendered seat instead of using a single inline placement", () => {
    const bubbleScene = structuredClone(mahjongRichScene) as VisualScene;
    const bubbleBody = bubbleScene.body as Record<string, any>;

    bubbleBody.panels = {
      ...(bubbleBody.panels as Record<string, any>),
      chatLog: [
        { playerId: "east", text: "east chat" },
        { playerId: "south", text: "south chat" },
        { playerId: "west", text: "west chat" },
        { playerId: "north", text: "north chat" },
      ],
    };

    render(
      <MahjongPlugin
        session={{
          sessionId: "mahjong-bubbles",
          gameId: "mahjong",
          pluginId: "arena.visualization.mahjong.table_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
            cursorTs: 2022,
            cursorEventSeq: 22,
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
        scene={bubbleScene}
        submitAction={vi.fn()}
        submitInput={vi.fn()}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByTestId("mahjong-seat-bottom-bubble")).toHaveClass("mahjong-seat__bubble--bottom");
    expect(screen.getByTestId("mahjong-seat-right-bubble")).toHaveClass("mahjong-seat__bubble--right");
    expect(screen.getByTestId("mahjong-seat-top-bubble")).toHaveClass("mahjong-seat__bubble--top");
    expect(screen.getByTestId("mahjong-seat-left-bubble")).toHaveClass("mahjong-seat__bubble--left");
  });

  it("prefers the session observer when replay payload carries another private view", () => {
    const replayScene = structuredClone(mahjongRichScene) as VisualScene;
    const replayBody = replayScene.body as Record<string, any>;
    const replayTable = replayBody.table as Record<string, any>;
    const replaySeats = replayTable.seats as Array<Record<string, any>>;

    replayBody.status = {
      ...(replayBody.status as Record<string, any>),
      activePlayerId: "south",
      observerPlayerId: null,
      privateViewPlayerId: "south",
    };
    replaySeats.forEach((seat) => {
      seat.isObserver = false;
      if (seat.playerId === "east") {
        seat.hand = {
          isVisible: false,
          cards: [],
          maskedCount: 13,
        };
      }
    });

    render(
      <MahjongPlugin
        session={{
          sessionId: "mahjong-replay",
          gameId: "mahjong",
          pluginId: "arena.visualization.mahjong.table_v1",
          lifecycle: "closed",
          playback: {
            mode: "paused",
            cursorTs: 2060,
            cursorEventSeq: 6,
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
            activeActorId: "south",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={replayScene}
        submitAction={vi.fn()}
        submitInput={vi.fn()}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByTestId("mahjong-stage-status")).toHaveTextContent("Viewing East");
    expect(screen.getByTestId("mahjong-stage-status")).not.toHaveTextContent("Viewing South");
    expect(screen.getByLabelText("Mahjong seat east")).toHaveAttribute(
      "data-testid",
      "mahjong-seat-bottom",
    );
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
