import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { VisualScene } from "../../gateway/types";
import doudizhuScene from "../../test/fixtures/doudizhu.visual.json";
import doudizhuRichScene from "../../test/fixtures/doudizhu.rich.visual";
import { DoudizhuActionComposer } from "./DoudizhuActionComposer";
import { DoudizhuPlugin } from "./DoudizhuPlugin";
import { DoudizhuCardVisual } from "./doudizhuCards";

function resolveSceneBody(scene: VisualScene): Record<string, any> {
  return scene.body as Record<string, any>;
}

describe("DoudizhuPlugin", () => {
  it("clears composed card selection when the hand changes without changing length", () => {
    const submitAction = vi.fn();
    const { rerender } = render(
      <DoudizhuActionComposer
        handCards={["C3", "C4"]}
        actionTexts={["3", "4"]}
        canSubmitActions={true}
        onSubmitAction={submitAction}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /select card c3/i }));
    expect(screen.getByText(/selected/i)).toHaveTextContent("C3");

    rerender(
      <DoudizhuActionComposer
        handCards={["C4", "C5"]}
        actionTexts={["4", "5"]}
        canSubmitActions={true}
        onSubmitAction={submitAction}
      />,
    );

    expect(screen.getByText("Select cards to compose a legal move.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /play selected/i })).toBeDisabled();
  });

  it("uses a three-row table layout so top seats, center cards, and the bottom hand stay separated", () => {
    const css = readFileSync(resolve(process.cwd(), "src/plugins/doudizhu/doudizhu.css"), "utf-8");

    expect(css).toMatch(
      /grid-template-areas:\s*"left right"\s*"center center"\s*"bottom bottom"/,
    );
    expect(css).toMatch(/\.doudizhu-center\s*\{[^}]*grid-area:\s*center/s);
    expect(css).toMatch(/\.doudizhu-seat--bottom\s*\{[^}]*grid-area:\s*bottom/s);
    expect(css).toMatch(/\.doudizhu-seat--left\s*\{[^}]*grid-area:\s*left/s);
    expect(css).toMatch(/\.doudizhu-seat--right\s*\{[^}]*grid-area:\s*right/s);
    expect(css).not.toMatch(/\.doudizhu-seat\s*\{[^}]*position:\s*absolute/s);
  });

  it("keeps the decorative table felt as a fixed geometry layer instead of a cover-scaled image", () => {
    const css = readFileSync(resolve(process.cwd(), "src/plugins/doudizhu/doudizhu.css"), "utf-8");

    expect(css).toMatch(/\.doudizhu-stage__table::before\s*\{/);
    expect(css).toMatch(/\.doudizhu-stage__table::before\s*\{[^}]*clip-path:\s*polygon/s);
    expect(css).toMatch(/\.doudizhu-stage__table::before\s*\{[^}]*transform:\s*translateX\(-50%\)/s);
    expect(css).toMatch(/\.doudizhu-stage__table::before\s*\{[^}]*height:\s*clamp\(/s);
    expect(css).not.toMatch(/\.doudizhu-stage__table\s*\{[^}]*url\("\.\/assets\/gameboard\.png"\)/s);
  });

  it("loads doudizhu portraits from arena-visual-owned assets instead of rlcard-showdown", () => {
    const source = readFileSync(resolve(process.cwd(), "src/plugins/doudizhu/DoudizhuTable.tsx"), "utf-8");

    expect(source).toMatch(/\.\/assets\/portraits\/\*\.\{png,jpg,jpeg\}/);
    expect(source).not.toMatch(/rlcard-showdown/);
  });

  it("loads baseline-style poker faces from plugin-local assets instead of drawing fake suit dots", () => {
    const source = readFileSync(resolve(process.cwd(), "src/plugins/doudizhu/doudizhuCards.tsx"), "utf-8");

    expect(source).toMatch(/\.\/assets\/cards\/\*\.\{png,jpg,jpeg\}/);
    expect(source).not.toMatch(/\|\|\s*"•"/);
  });

  it("uses a shared click handler for interactive hand cards", () => {
    const source = readFileSync(resolve(process.cwd(), "src/plugins/doudizhu/DoudizhuActionComposer.tsx"), "utf-8");

    expect(source).toMatch(/onClick=\{canSubmitActions \? handleCardClick : undefined\}/);
    expect(source).not.toMatch(/onClick=\{\s*canSubmitActions\s*\?\s*\(\)\s*=>/s);
  });

  it("renders suitful 10 cards with baseline card images instead of transparent text fallbacks", () => {
    const { container, rerender } = render(<DoudizhuCardVisual card="CT" />);

    expect(container.querySelector(".doudizhu-card__image")).not.toBeNull();
    expect(container.querySelector(".doudizhu-card__rank")).toBeNull();
    expect(screen.getByLabelText("Doudizhu card CT")).toBeInTheDocument();

    rerender(<DoudizhuCardVisual card="ST" />);

    expect(container.querySelector(".doudizhu-card__image")).not.toBeNull();
    expect(container.querySelector(".doudizhu-card__rank")).toBeNull();
    expect(screen.getByLabelText("Doudizhu card ST")).toBeInTheDocument();
  });

  it("does not render a fake facedown card when a hidden hand has zero masked cards", () => {
    const baseBody = resolveSceneBody(doudizhuScene as VisualScene);
    const baseTable = baseBody.table as Record<string, any>;
    const scene = {
      ...(doudizhuScene as VisualScene),
      body: {
        ...baseBody,
        table: {
          ...baseTable,
          seats: (baseTable.seats as Array<Record<string, any>>).map((seat) =>
            seat.seatId === "left"
              ? {
                  ...seat,
                  hand: {
                    ...seat.hand,
                    maskedCount: 0,
                  },
                }
              : seat,
          ),
        },
      },
    } satisfies VisualScene;

    render(
      <DoudizhuPlugin
        session={{
          sessionId: "doudizhu-zero-mask",
          gameId: "doudizhu",
          pluginId: "arena.visualization.doudizhu.table_v1",
          lifecycle: "live_running",
          playback: {
            mode: "paused",
            cursorTs: 1007,
            cursorEventSeq: 7,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "player_0",
            observerKind: "player",
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={scene}
        submitAction={vi.fn()}
        submitInput={vi.fn()}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByTestId("doudizhu-seat-left-hand")).toHaveTextContent("Hidden hand");
    expect(
      screen
        .getByTestId("doudizhu-seat-left-hand")
        .querySelectorAll(".doudizhu-hand__cards > .doudizhu-card--back"),
    ).toHaveLength(0);
  });

  it("renders a plugin-local doudizhu stage, masks other seats, and submits action controls", () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);
    const baseBody = resolveSceneBody(doudizhuScene as VisualScene);
    const scene = {
      ...(doudizhuScene as VisualScene),
      body: {
        ...baseBody,
        status: {
          ...(baseBody.status as Record<string, unknown>),
          privateViewPlayerId: "player_0",
        },
      },
    } satisfies VisualScene;

    render(
      <DoudizhuPlugin
        session={{
          sessionId: "doudizhu-sample",
          gameId: "doudizhu",
          pluginId: "arena.visualization.doudizhu.table_v1",
          lifecycle: "live_running",
          playback: {
            mode: "paused",
            cursorTs: 1007,
            cursorEventSeq: 7,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "player_0",
            observerKind: "player",
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={scene}
        submitAction={vi.fn()}
        submitInput={submitInput}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByTestId("doudizhu-stage")).toBeInTheDocument();
    expect(screen.getByTestId("doudizhu-center-cards")).toHaveTextContent("♥");
    expect(screen.getByTestId("doudizhu-center-cards")).toHaveTextContent("♠");
    expect(screen.getByTestId("doudizhu-seat-bottom")).toHaveTextContent("Player 0");
    expect(screen.getByTestId("doudizhu-seat-bottom")).toHaveTextContent("landlord");
    expect(screen.getByTestId("doudizhu-seat-bottom-hand")).toHaveTextContent("♥");
    expect(screen.getByTestId("doudizhu-seat-bottom-hand")).toHaveTextContent("♠");
    expect(screen.getByRole("button", { name: /select card blackjoker/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /select card redjoker/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /show legal actions/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /play legal 4/i })).toBeNull();
    expect(screen.getByTestId("doudizhu-seat-left-hand")).toHaveTextContent("Hidden hand");
    expect(screen.getByTestId("doudizhu-seat-left-hand")).toHaveTextContent("17 cards");
    expect(screen.getByTestId("doudizhu-seat-right-hand")).toHaveTextContent("Hidden hand");
    expect(screen.getByTestId("doudizhu-seat-right-hand")).toHaveTextContent("17 cards");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Move 1");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Turn Player 0");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Last move 3");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Landlord Player 0");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Viewing Player 0");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Recent plays");
    expect(screen.getByTestId("doudizhu-seat-left")).toHaveTextContent("No public cards");
    expect(
      screen
        .getByTestId("doudizhu-seat-bottom-hand")
        .querySelectorAll(".doudizhu-hand__cards > .doudizhu-card"),
    ).toHaveLength(17);
    expect(
      screen
        .getByTestId("doudizhu-seat-left-hand")
        .querySelectorAll(".doudizhu-hand__cards > .doudizhu-card--back"),
    ).toHaveLength(17);
    expect(
      screen
        .getByTestId("doudizhu-seat-right-hand")
        .querySelectorAll(".doudizhu-hand__cards > .doudizhu-card--back"),
    ).toHaveLength(17);

    expect(screen.getByRole("button", { name: /hint/i })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: /show legal actions/i }));
    expect(screen.getByRole("button", { name: /play legal 4/i })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^pass$/i }));
    expect(submitInput).toHaveBeenCalledWith({
      playerId: "player_0",
      actionText: "pass",
    });
  });

  it("keeps host-owned summary and chat panels out of the plugin content block", () => {
    render(
      <DoudizhuPlugin
        session={{
          sessionId: "doudizhu-sample",
          gameId: "doudizhu",
          pluginId: "arena.visualization.doudizhu.table_v1",
          lifecycle: "live_running",
          playback: {
            mode: "paused",
            cursorTs: 1007,
            cursorEventSeq: 7,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "player_0",
            observerKind: "player",
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={doudizhuScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={vi.fn()}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByLabelText("Doudizhu seat bottom")).toBeInTheDocument();
    expect(screen.getByLabelText("Doudizhu seat left")).toBeInTheDocument();
    expect(screen.getByLabelText("Doudizhu seat right")).toBeInTheDocument();
    expect(screen.getAllByLabelText(/Doudizhu card /i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/^History$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Table talk$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/observer: player_0/i)).not.toBeInTheDocument();
  });

  it("lets the human compose a legal move from selected hand cards instead of relying on raw action chips", () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);

    render(
      <DoudizhuPlugin
        session={{
          sessionId: "doudizhu-sample",
          gameId: "doudizhu",
          pluginId: "arena.visualization.doudizhu.table_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
            cursorTs: 1016,
            cursorEventSeq: 16,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "player_0",
            observerKind: "player",
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={doudizhuRichScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={submitInput}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByRole("button", { name: /hint/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /play selected/i })).toBeDisabled();
    expect(screen.queryByRole("button", { name: /play legal r/i })).toBeNull();
    expect(screen.getByTestId("doudizhu-seat-bottom-hand")).toHaveTextContent("♥");
    expect(screen.getByTestId("doudizhu-seat-bottom-hand")).toHaveTextContent("♠");
    expect(screen.getByTestId("doudizhu-seat-bottom")).toHaveTextContent("♥");

    fireEvent.click(screen.getByRole("button", { name: /show legal actions/i }));
    expect(screen.getByRole("button", { name: /play legal r/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /select card s6/i }));
    fireEvent.click(screen.getByRole("button", { name: /select card hj/i }));
    fireEvent.click(screen.getByRole("button", { name: /select card cj/i }));
    fireEvent.click(screen.getByRole("button", { name: /select card dj/i }));

    expect(submitInput).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /play 6jjj/i })).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: /play 6jjj/i }));
    expect(submitInput).toHaveBeenCalledWith({
      playerId: "player_0",
      actionText: "6JJJ",
    });
  });

  it("shows the latest seat chat as local table bubbles without restoring a second chat panel", () => {
    render(
      <DoudizhuPlugin
        session={{
          sessionId: "doudizhu-sample",
          gameId: "doudizhu",
          pluginId: "arena.visualization.doudizhu.table_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
            cursorTs: 1016,
            cursorEventSeq: 16,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "player_0",
            observerKind: "player",
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={doudizhuRichScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={vi.fn()}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByTestId("doudizhu-seat-bottom")).toHaveTextContent("I have rocket");
    expect(screen.getByTestId("doudizhu-seat-left")).toHaveTextContent("last warning");
    expect(screen.queryByText("watch this")).toBeNull();
    expect(screen.queryByText(/^Table talk$/i)).not.toBeInTheDocument();
  });

  it("anchors seat chat bubbles to the portrait block like the release baseline", () => {
    const css = readFileSync(resolve(process.cwd(), "src/plugins/doudizhu/doudizhu.css"), "utf-8");

    render(
      <DoudizhuPlugin
        session={{
          sessionId: "doudizhu-sample",
          gameId: "doudizhu",
          pluginId: "arena.visualization.doudizhu.table_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
            cursorTs: 1016,
            cursorEventSeq: 16,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "player_0",
            observerKind: "player",
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={doudizhuRichScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={vi.fn()}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    const bottomSeat = screen.getByTestId("doudizhu-seat-bottom");
    const bottomIdentity = bottomSeat.querySelector(".doudizhu-seat__identity");
    const bottomBubble = screen.getByTestId("doudizhu-seat-bottom-bubble");

    expect(bottomIdentity).not.toBeNull();
    expect(bottomIdentity).toContainElement(bottomBubble);
    expect(css).toMatch(/\.doudizhu-seat__identity\s*\{[^}]*position:\s*relative;/s);
    expect(css).toMatch(/\.doudizhu-seat__bubble\s*\{[^}]*position:\s*absolute;/s);
  });

  it("surfaces rich live-table status cues without restoring a host-owned history shell", () => {
    const richBody = resolveSceneBody(doudizhuRichScene as VisualScene);
    const scene = {
      ...doudizhuRichScene,
      body: {
        ...richBody,
        status: {
          ...(richBody.status as Record<string, unknown>),
          privateViewPlayerId: "player_0",
        },
      },
    } satisfies VisualScene;

    render(
      <DoudizhuPlugin
        session={{
          sessionId: "doudizhu-sample",
          gameId: "doudizhu",
          pluginId: "arena.visualization.doudizhu.table_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
            cursorTs: 1016,
            cursorEventSeq: 16,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "player_0",
            observerKind: "player",
          },
          scheduling: {
            family: "turn",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={scene}
        submitAction={vi.fn()}
        submitInput={vi.fn()}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Move 9");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Last move 10");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Turn Player 0");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Landlord Player 0");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Viewing Player 0");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Recent plays");
    expect(screen.getByTestId("doudizhu-center-cards")).toHaveTextContent("♠");
    expect(screen.getByTestId("doudizhu-center-cards")).toHaveTextContent("♥");
    expect(screen.queryByText(/^History$/i)).not.toBeInTheDocument();
  });

  it("lets side-seat recent public cards wrap across multiple rows instead of forcing a single line", () => {
    const css = readFileSync(resolve(process.cwd(), "src/plugins/doudizhu/doudizhu.css"), "utf-8");

    expect(css).toMatch(/\.doudizhu-seat__played,\s*\.doudizhu-seat__hand\s*\{[^}]*display:\s*grid/s);
    expect(css).toMatch(/\.doudizhu-played,\s*\.doudizhu-hand__cards\s*\{[^}]*flex-wrap:\s*wrap/s);
  });

  it("keeps masked hidden hands fully visible instead of clipping the back-card fan", () => {
    const css = readFileSync(resolve(process.cwd(), "src/plugins/doudizhu/doudizhu.css"), "utf-8");

    expect(css).toMatch(/\.doudizhu-hand__cards--masked\s*\{[^}]*overflow:\s*visible/s);
  });

  it("keeps camera observer views masked and read-only for the doudizhu table", () => {
    const baseBody = resolveSceneBody(doudizhuScene as VisualScene);
    const baseTable = baseBody.table as Record<string, any>;
    const cameraScene = {
      ...(doudizhuScene as VisualScene),
      body: {
        ...baseBody,
        table: {
          ...baseTable,
          seats: [
            {
              seatId: "bottom",
              playerId: "player_0",
              playerName: "Player 0",
              role: "landlord",
              isActive: true,
              isObserver: false,
              playedCards: ["3"],
              publicNotes: [],
              hand: {
                isVisible: false,
                cards: [],
                maskedCount: 17,
              },
            },
            ...(baseTable.seats as unknown[]).slice(1),
          ],
        },
        status: {
          ...(baseBody.status as Record<string, unknown>),
          observerPlayerId: null,
          privateViewPlayerId: "player_0",
        },
      },
    } satisfies VisualScene;

    render(
      <DoudizhuPlugin
        session={{
          sessionId: "doudizhu-camera",
          gameId: "doudizhu",
          pluginId: "arena.visualization.doudizhu.table_v1",
          lifecycle: "live_running",
          playback: {
            mode: "paused",
            cursorTs: 1007,
            cursorEventSeq: 7,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "",
            observerKind: "camera",
          },
          scheduling: {
            family: "turn",
            phase: "idle",
            acceptsHumanIntent: false,
            activeActorId: "player_0",
          },
          capabilities: {
            observerModes: ["global", "spectator", "camera", "player"],
          },
          summary: {},
          timeline: {},
        }}
        scene={cameraScene}
        submitAction={vi.fn()}
        submitInput={vi.fn()}
        mediaSubscribe={() => () => {}}
        isFallback={false}
      />,
    );

    expect(screen.getByTestId("doudizhu-seat-bottom-hand")).toHaveTextContent("Hidden hand");
    expect(screen.getByTestId("doudizhu-seat-bottom-hand")).toHaveTextContent("17 cards");
    expect(
      screen
        .getByTestId("doudizhu-seat-bottom-hand")
        .querySelectorAll(".doudizhu-hand__cards > .doudizhu-card--back"),
    ).toHaveLength(17);
    expect(screen.queryByRole("button", { name: /show legal actions/i })).toBeNull();
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Viewing Player 0");
  });
});
