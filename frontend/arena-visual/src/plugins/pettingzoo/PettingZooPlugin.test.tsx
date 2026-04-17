import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ResolvedMediaSource } from "../../gateway/media";
import type { VisualScene } from "../../gateway/types";
import pettingzooScene from "../../test/fixtures/pettingzoo.visual.json";
import { PettingZooPlugin } from "./PettingZooPlugin";

const FRAME_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAGUlEQVR4nGNkaGBgYGBg+M8ABYwMjAyMDAwAAB0vAQx0J7s8AAAAAElFTkSuQmCC";
const FRAME_DATA_URL_2 =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAHElEQVR4nGNkYPjPAAJMDKiAVTAxoKJAAgA+MAEF4c6mRwAAAABJRU5ErkJggg==";

describe("PettingZooPlugin", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders an inline frame immediately when the scene already carries a data-url media ref", async () => {
    render(
      <PettingZooPlugin
        session={{
          sessionId: "pettingzoo-sample",
          gameId: "pettingzoo",
          pluginId: "arena.visualization.pettingzoo.frame_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
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
        scene={{
          ...(pettingzooScene as VisualScene),
          media: {
            primary: {
              mediaId: "inline-media-1",
              transport: "http_pull",
              mimeType: "image/png",
              url: FRAME_DATA_URL
            }
          }
        }}
        submitAction={async () => undefined}
        submitInput={async () => undefined}
        mediaSubscribe={(request, listener) => {
          listener({
            mediaId: request.mediaId,
            status: "loading"
          } as ResolvedMediaSource);
          return () => {};
        }}
        isFallback={false}
      />,
    );

    const image = await screen.findByTestId("frame-surface-image");
    expect(image).toHaveAttribute("src", FRAME_DATA_URL);
    expect(screen.queryByText(/loading frame/i)).not.toBeInTheDocument();
  });

  it("updates visible frame status when scrubbing to a new scene", async () => {
    const { container, rerender } = render(
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
    expect(container.querySelector(".frame-surface__overlay-strip")).toBeNull();
  });

  it("prefers the current scene inline media over a stale subscribed frame while the next media id is loading", async () => {
    const listeners = new Map<string, (state: ResolvedMediaSource) => void>();
    const subscribe = (request: { mediaId: string }, listener: (state: ResolvedMediaSource) => void) => {
      listeners.set(request.mediaId, listener);
      if (request.mediaId === "frame-1") {
        listener({
          mediaId: request.mediaId,
          status: "ready",
          src: FRAME_DATA_URL
        } as ResolvedMediaSource);
      } else {
        listener({
          mediaId: request.mediaId,
          status: "loading"
        } as ResolvedMediaSource);
      }
      return () => {
        listeners.delete(request.mediaId);
      };
    };

    const { rerender } = render(
      <PettingZooPlugin
        session={{
          sessionId: "pettingzoo-sample",
          gameId: "pettingzoo",
          pluginId: "arena.visualization.pettingzoo.frame_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
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
        scene={{
          ...(pettingzooScene as VisualScene),
          media: {
            primary: {
              mediaId: "frame-1",
              transport: "http_pull",
              mimeType: "image/png",
              url: FRAME_DATA_URL
            }
          }
        }}
        submitAction={async () => undefined}
        submitInput={async () => undefined}
        mediaSubscribe={subscribe}
        isFallback={false}
      />,
    );

    await waitFor(() =>
      expect(screen.getByTestId("frame-surface-image")).toHaveAttribute("src", FRAME_DATA_URL),
    );

    rerender(
      <PettingZooPlugin
        session={{
          sessionId: "pettingzoo-sample",
          gameId: "pettingzoo",
          pluginId: "arena.visualization.pettingzoo.frame_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
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
            activeActorId: "pilot_0"
          },
          capabilities: {},
          summary: {},
          timeline: {}
        }}
        scene={{
          ...(pettingzooScene as VisualScene),
          seq: 12,
          media: {
            primary: {
              mediaId: "frame-2",
              transport: "http_pull",
              mimeType: "image/png",
              url: FRAME_DATA_URL_2
            }
          }
        }}
        submitAction={async () => undefined}
        submitInput={async () => undefined}
        mediaSubscribe={subscribe}
        isFallback={false}
      />,
    );

    await waitFor(() =>
      expect(screen.getByTestId("frame-surface-image")).toHaveAttribute("src", FRAME_DATA_URL_2),
    );
  });

  it("throttles repeated Space Invaders FIRE keyboard submissions for 500ms", async () => {
    let nowMs = 0;
    vi.spyOn(performance, "now").mockImplementation(() => nowMs);
    const submitInput = vi.fn().mockResolvedValue(undefined);

    render(
      <PettingZooPlugin
        session={{
          sessionId: "pettingzoo-sample",
          gameId: "pettingzoo",
          pluginId: "arena.visualization.pettingzoo.frame_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
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
        submitInput={submitInput}
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

    await waitFor(() =>
      expect(screen.getByTestId("frame-surface-image")).toHaveAttribute("src", FRAME_DATA_URL),
    );

    fireEvent.keyDown(window, { key: " " });
    fireEvent.keyUp(window, { key: " " });
    nowMs = 200;
    fireEvent.keyDown(window, { key: " " });
    fireEvent.keyUp(window, { key: " " });

    expect(submitInput).toHaveBeenCalledTimes(1);
    expect(submitInput).toHaveBeenLastCalledWith({
      playerId: "pilot_0",
      actionPayload: expect.objectContaining({
        id: "FIRE",
        move: "FIRE",
        metadata: expect.objectContaining({
          realtime_input: true
        })
      })
    });

    nowMs = 501;
    fireEvent.keyDown(window, { key: " " });

    expect(submitInput).toHaveBeenCalledTimes(2);
  });
});
