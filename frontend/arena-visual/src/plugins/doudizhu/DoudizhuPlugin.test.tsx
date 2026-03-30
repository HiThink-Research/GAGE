import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { VisualScene } from "../../gateway/types";
import doudizhuScene from "../../test/fixtures/doudizhu.visual.json";
import doudizhuRichScene from "../../test/fixtures/doudizhu.rich.visual";
import { DoudizhuPlugin } from "./DoudizhuPlugin";

function resolveSceneBody(scene: VisualScene): Record<string, any> {
  return scene.body as Record<string, any>;
}

describe("DoudizhuPlugin", () => {
  it("uses a wider grid table layout so richer hands can spread across the stage", () => {
    const css = readFileSync(resolve(process.cwd(), "src/plugins/doudizhu/doudizhu.css"), "utf-8");

    expect(css).toMatch(
      /grid-template-areas:\s*"left center right"\s*"bottom bottom bottom"/,
    );
    expect(css).toMatch(/\.doudizhu-center\s*\{[^}]*grid-area:\s*center/s);
    expect(css).toMatch(/\.doudizhu-seat--bottom\s*\{[^}]*grid-area:\s*bottom/s);
    expect(css).not.toMatch(/\.doudizhu-seat\s*\{[^}]*position:\s*absolute/s);
  });

  it("loads doudizhu portraits from arena-visual-owned assets instead of rlcard-showdown", () => {
    const source = readFileSync(resolve(process.cwd(), "src/plugins/doudizhu/DoudizhuTable.tsx"), "utf-8");

    expect(source).toMatch(/\.\/assets\/portraits\/\*\.\{png,jpg,jpeg\}/);
    expect(source).not.toMatch(/rlcard-showdown/);
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
    expect(screen.getByTestId("doudizhu-center-cards")).toHaveTextContent("3");
    expect(screen.getByTestId("doudizhu-seat-bottom")).toHaveTextContent("Player 0");
    expect(screen.getByTestId("doudizhu-seat-bottom")).toHaveTextContent("landlord");
    expect(screen.getByTestId("doudizhu-seat-bottom-hand")).toHaveTextContent("3");
    expect(screen.getByTestId("doudizhu-seat-bottom-hand")).toHaveTextContent("4");
    expect(screen.getByRole("button", { name: /select card blackjoker/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /select card redjoker/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /show legal actions/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /play legal 4/i })).toBeNull();
    expect(screen.getByTestId("doudizhu-seat-left-hand")).toHaveTextContent("Hidden hand");
    expect(screen.getByTestId("doudizhu-seat-right-hand")).toHaveTextContent("Hidden hand");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Move 1");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Turn Player 0");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Last move 3");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Landlord Player 0");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Viewing Player 0");
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Recent plays");
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
    expect(screen.getAllByLabelText("Doudizhu card 3").length).toBeGreaterThan(0);
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
    expect(screen.queryByText(/^History$/i)).not.toBeInTheDocument();
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
    expect(
      screen
        .getByTestId("doudizhu-seat-bottom-hand")
        .querySelectorAll(".doudizhu-hand__cards > .doudizhu-card--back"),
    ).toHaveLength(17);
    expect(screen.queryByRole("button", { name: /show legal actions/i })).toBeNull();
    expect(screen.getByTestId("doudizhu-stage")).toHaveTextContent("Viewing Player 0");
  });
});
