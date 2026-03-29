import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { ResolvedMediaSource } from "../../gateway/media";
import type { VisualScene } from "../../gateway/types";
import pettingzooScene from "../../test/fixtures/pettingzoo.visual.json";
import { PettingZooPlugin } from "./PettingZooPlugin";

const FRAME_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAGUlEQVR4nGNkaGBgYGBg+M8ABYwMjAyMDAwAAB0vAQx0J7s8AAAAAElFTkSuQmCC";

describe("PettingZooPlugin", () => {
  it("updates visible frame status when scrubbing to a new scene", async () => {
    const { rerender } = render(
      <PettingZooPlugin
        session={{
          sessionId: "pettingzoo-sample",
          gameId: "pettingzoo",
          pluginId: "arena.visualization.pettingzoo.frame_v1",
          lifecycle: "closed",
          playback: {
            mode: "paused",
            cursorTs: 2011,
            cursorEventSeq: 11,
            speed: 1,
            canSeek: true
          },
          observer: {
            observerId: "pilot_0",
            observerKind: "player"
          },
          scheduling: {
            family: "agent_cycle",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "pilot_0"
          },
          capabilities: {},
          summary: {},
          timeline: {}
        }}
        scene={pettingzooScene as VisualScene}
        submitAction={async () => undefined}
        submitInput={async () => undefined}
        mediaSubscribe={(request, listener) => {
          listener({
            mediaId: request.mediaId,
            status: "ready",
            src: FRAME_DATA_URL
          } as ResolvedMediaSource);
          return () => {};
        }}
        isFallback={false}
      />,
    );

    await waitFor(() => expect(screen.getByTestId("frame-status-line")).toHaveTextContent("Tick 3"));
    expect(screen.getByTestId("frame-surface-viewport")).toHaveStyle("width: min(100%, 26rem)");
    expect(screen.getByText("Space Invaders wave 3")).toBeInTheDocument();

    rerender(
      <PettingZooPlugin
        session={{
          sessionId: "pettingzoo-sample",
          gameId: "pettingzoo",
          pluginId: "arena.visualization.pettingzoo.frame_v1",
          lifecycle: "closed",
          playback: {
            mode: "paused",
            cursorTs: 2012,
            cursorEventSeq: 12,
            speed: 1,
            canSeek: true
          },
          observer: {
            observerId: "pilot_0",
            observerKind: "player"
          },
          scheduling: {
            family: "agent_cycle",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "pilot_1"
          },
          capabilities: {},
          summary: {},
          timeline: {}
        }}
        scene={{
          ...(pettingzooScene as VisualScene),
          seq: 12,
          body: {
            ...(pettingzooScene as VisualScene).body as Record<string, unknown>,
            status: {
              activePlayerId: "pilot_1",
              observerPlayerId: "pilot_0",
              tick: 4,
              step: 4,
              moveCount: 4,
              lastMove: "LEFT",
              reward: 2
            }
          },
          overlays: [
            { kind: "badge", label: "Tick", value: "4" },
            { kind: "badge", label: "Reward", value: "2.0" },
            { kind: "badge", label: "Last move", value: "LEFT" }
          ]
        }}
        submitAction={async () => undefined}
        submitInput={async () => undefined}
        mediaSubscribe={(request, listener) => {
          listener({
            mediaId: request.mediaId,
            status: "ready",
            src: FRAME_DATA_URL
          } as ResolvedMediaSource);
          return () => {};
        }}
        isFallback={false}
      />,
    );

    await waitFor(() => expect(screen.getByTestId("frame-status-line")).toHaveTextContent("Tick 4"));
    expect(screen.getByTestId("frame-status-line")).toHaveTextContent("LEFT");
  });
});
