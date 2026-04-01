import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ResolvedMediaSource } from "../../gateway/media";
import type { VisualScene } from "../../gateway/types";
import retroScene from "../../test/fixtures/retro-mario.visual.json";
import { RetroMarioPlugin } from "./RetroMarioPlugin";

const FRAME_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAGUlEQVR4nGNkaGBgYGBg+M8ABYwMjAyMDAwAAB0vAQx0J7s8AAAAAElFTkSuQmCC";

describe("RetroMarioPlugin", () => {
  it("renders low-latency Mario frames through a single multipart stream reader instead of polling content snapshots", async () => {
    const encoder = new TextEncoder();
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(
              encoder.encode(
                "--frame\r\nContent-Type: image/png\r\nContent-Length: 11\r\n\r\n",
              ),
            );
            controller.enqueue(encoder.encode("frame-bytes"));
            controller.enqueue(encoder.encode("\r\n--frame--\r\n"));
            controller.close();
          },
        }),
        {
          status: 200,
          headers: {
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
          },
        },
      ),
    );
    const createImageBitmapMock = vi.fn().mockResolvedValue({
      width: 4,
      height: 4,
      close: vi.fn(),
    });
    const drawImage = vi.fn();
    const getContextMock = vi
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockReturnValue({ drawImage, clearRect: vi.fn() } as unknown as CanvasRenderingContext2D);
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("createImageBitmap", createImageBitmapMock);

    const lowLatencyScene = JSON.parse(JSON.stringify(retroScene)) as VisualScene;
    if (lowLatencyScene.media?.primary) {
      lowLatencyScene.media.primary.mediaId = "live-channel-main";
      lowLatencyScene.media.primary.transport = "low_latency_channel";
      lowLatencyScene.media.primary.mimeType = "multipart/x-mixed-replace";
      lowLatencyScene.media.primary.url =
        "/arena_visual/sessions/retro-sample/media/live-channel-main/stream?run_id=run-live";
    }

    try {
      render(
        <RetroMarioPlugin
          session={{
            sessionId: "retro-sample",
            gameId: "retro_platformer",
            pluginId: "arena.visualization.retro_platformer.frame_v1",
            lifecycle: "live_running",
            playback: {
              mode: "live_tail",
              cursorTs: 4023,
              cursorEventSeq: 23,
              speed: 1,
              canSeek: true
            },
            observer: {
              observerId: "player_0",
              observerKind: "player"
            },
            scheduling: {
              family: "real_time_tick",
              phase: "waiting_for_intent",
              acceptsHumanIntent: true,
              activeActorId: "player_0"
            },
            capabilities: {},
            summary: {},
            timeline: {}
          }}
          scene={lowLatencyScene}
          submitAction={vi.fn()}
          submitInput={vi.fn()}
          mediaSubscribe={(request, listener) => {
            listener({
              mediaId: request.mediaId,
              status: "ready",
              src: "http://arena.local/arena_visual/sessions/retro-sample/media/live-channel-main/stream?run_id=run-live",
              ref: {
                mediaId: request.mediaId,
                transport: "low_latency_channel",
                mimeType: "multipart/x-mixed-replace",
                url: "/arena_visual/sessions/retro-sample/media/live-channel-main/stream?run_id=run-live"
              }
            } as ResolvedMediaSource);
            return () => {};
          }}
          isFallback={false}
        />,
      );

      await waitFor(() =>
        expect(screen.getByTestId("frame-surface-canvas")).toBeInTheDocument(),
      );
      await waitFor(() =>
        expect(fetchMock).toHaveBeenCalled(),
      );
      expect(String(fetchMock.mock.calls[0]?.[0] ?? "")).toContain(
        "/arena_visual/sessions/retro-sample/media/live-channel-main/stream?run_id=run-live",
      );
      expect(fetchMock.mock.calls[0]?.[1]).toEqual(
        expect.objectContaining({ cache: "no-store" }),
      );
      expect(screen.queryByTestId("frame-surface-image")).not.toBeInTheDocument();
      expect(drawImage).toHaveBeenCalled();
    } finally {
      getContextMock.mockRestore();
      vi.unstubAllGlobals();
    }
  });

  it("renders Mario in immersive mode without frame metadata chrome or action chips", async () => {
    render(
      <RetroMarioPlugin
        session={{
          sessionId: "retro-sample",
          gameId: "retro_platformer",
          pluginId: "arena.visualization.retro_platformer.frame_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
            cursorTs: 4023,
            cursorEventSeq: 23,
            speed: 1,
            canSeek: true
          },
          observer: {
            observerId: "player_0",
            observerKind: "player"
          },
          scheduling: {
            family: "real_time_tick",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0"
          },
          capabilities: {},
          summary: {},
          timeline: {}
        }}
        scene={retroScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={vi.fn()}
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

    expect(screen.getByTestId("frame-surface-root")).toHaveClass("frame-surface--immersive");
    expect(screen.queryByText("Retro Mario Frame")).not.toBeInTheDocument();
    expect(screen.queryByText("Tick 23")).not.toBeInTheDocument();
    expect(screen.queryByTestId("frame-status-line")).not.toBeInTheDocument();
    expect(screen.queryByTestId("frame-keyboard-hint")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /right \+ jump/i })).not.toBeInTheDocument();
  });

  it("renders the live frame after starting from an unavailable scene", async () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);
    const { rerender } = render(
      <RetroMarioPlugin
        session={{
          sessionId: "retro-sample",
          gameId: "retro_platformer",
          pluginId: "arena.visualization.retro_platformer.frame_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
            cursorTs: 4023,
            cursorEventSeq: 23,
            speed: 1,
            canSeek: true
          },
          observer: {
            observerId: "player_0",
            observerKind: "player"
          },
          scheduling: {
            family: "real_time_tick",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0"
          },
          capabilities: {},
          summary: {},
          timeline: {}
        }}
        scene={undefined}
        submitAction={vi.fn()}
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

    expect(screen.getByText(/frame unavailable/i)).toBeInTheDocument();

    rerender(
      <RetroMarioPlugin
        session={{
          sessionId: "retro-sample",
          gameId: "retro_platformer",
          pluginId: "arena.visualization.retro_platformer.frame_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
            cursorTs: 4023,
            cursorEventSeq: 23,
            speed: 1,
            canSeek: true
          },
          observer: {
            observerId: "player_0",
            observerKind: "player"
          },
          scheduling: {
            family: "real_time_tick",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0"
          },
          capabilities: {},
          summary: {},
          timeline: {}
        }}
        scene={retroScene as VisualScene}
        submitAction={vi.fn()}
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
  });

  it("cleans up the media subscription when the frame surface unmounts", () => {
    const unsubscribe = vi.fn();

    const { unmount } = render(
      <RetroMarioPlugin
        session={{
          sessionId: "retro-sample",
          gameId: "retro_platformer",
          pluginId: "arena.visualization.retro_platformer.frame_v1",
          lifecycle: "closed",
          playback: {
            mode: "paused",
            cursorTs: 4023,
            cursorEventSeq: 23,
            speed: 1,
            canSeek: true
          },
          observer: {
            observerId: "player_0",
            observerKind: "player"
          },
          scheduling: {
            family: "real_time_tick",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0"
          },
          capabilities: {},
          summary: {},
          timeline: {}
        }}
        scene={retroScene as VisualScene}
        submitAction={async () => undefined}
        submitInput={async () => undefined}
        mediaSubscribe={(request, listener) => {
          listener({
            mediaId: request.mediaId,
            status: "ready",
            src: FRAME_DATA_URL
          } as ResolvedMediaSource);
          return unsubscribe;
        }}
        isFallback={false}
      />,
    );

    unmount();
    expect(unsubscribe).toHaveBeenCalledTimes(1);
  });

  it("maps keyboard state into retro actions without resubmitting held keys on every new frame", async () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);
    const scene = JSON.parse(JSON.stringify(retroScene)) as VisualScene;
    const { rerender } = render(
      <RetroMarioPlugin
        session={{
          sessionId: "retro-sample",
          gameId: "retro_platformer",
          pluginId: "arena.visualization.retro_platformer.frame_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
            cursorTs: 4023,
            cursorEventSeq: 23,
            speed: 1,
            canSeek: true
          },
          observer: {
            observerId: "player_0",
            observerKind: "player"
          },
          scheduling: {
            family: "real_time_tick",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0"
          },
          capabilities: {},
          summary: {},
          timeline: {}
        }}
        scene={scene}
        submitAction={vi.fn()}
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

    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(submitInput).toHaveBeenLastCalledWith({
      playerId: "player_0",
      actionPayload: expect.objectContaining({
        id: "right",
        move: "right",
        hold_ticks: 6,
        metadata: expect.objectContaining({
          input_seq: 1,
          realtime_input: true
        })
      })
    });

    const nextScene = JSON.parse(JSON.stringify(scene)) as VisualScene;
    nextScene.sceneId = "retro-sample:seq:24";
    nextScene.seq = 24;
    nextScene.tsMs = 4024;
    (nextScene.body as { status: { tick: number } }).status.tick = 24;
    rerender(
      <RetroMarioPlugin
        session={{
          sessionId: "retro-sample",
          gameId: "retro_platformer",
          pluginId: "arena.visualization.retro_platformer.frame_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
            cursorTs: 4024,
            cursorEventSeq: 24,
            speed: 1,
            canSeek: true
          },
          observer: {
            observerId: "player_0",
            observerKind: "player"
          },
          scheduling: {
            family: "real_time_tick",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0"
          },
          capabilities: {},
          summary: {},
          timeline: {}
        }}
        scene={nextScene}
        submitAction={vi.fn()}
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
    expect(submitInput).toHaveBeenCalledTimes(1);

    fireEvent.keyDown(window, { key: " " });
    expect(submitInput).toHaveBeenLastCalledWith({
      playerId: "player_0",
      actionPayload: expect.objectContaining({
        id: "right_jump",
        move: "right_jump",
        hold_ticks: 6,
        metadata: expect.objectContaining({
          input_seq: 2,
          realtime_input: true
        })
      })
    });

    fireEvent.keyUp(window, { key: " " });
    await waitFor(() =>
      expect(submitInput).toHaveBeenLastCalledWith({
        playerId: "player_0",
        actionPayload: expect.objectContaining({
          id: "right",
          move: "right",
          hold_ticks: 6,
          metadata: expect.objectContaining({
            input_seq: 3,
            realtime_input: true
          })
        })
      }),
    );

    fireEvent.keyUp(window, { key: "ArrowRight" });
    await waitFor(() =>
      expect(submitInput).toHaveBeenLastCalledWith({
        playerId: "player_0",
        actionPayload: expect.objectContaining({
          id: "noop",
          move: "noop",
          metadata: expect.objectContaining({
            input_seq: 4,
            realtime_input: true
          })
        })
      }),
    );
  });

  it("submits new keyboard state immediately even while a prior realtime submit is still in flight", async () => {
    const submitInput = vi.fn().mockImplementation(() => new Promise<void>(() => {}));

    render(
      <RetroMarioPlugin
        session={{
          sessionId: "retro-sample",
          gameId: "retro_platformer",
          pluginId: "arena.visualization.retro_platformer.frame_v1",
          lifecycle: "live_running",
          playback: {
            mode: "live_tail",
            cursorTs: 4023,
            cursorEventSeq: 23,
            speed: 1,
            canSeek: true
          },
          observer: {
            observerId: "player_0",
            observerKind: "player"
          },
          scheduling: {
            family: "real_time_tick",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0"
          },
          capabilities: {},
          summary: {},
          timeline: {}
        }}
        scene={retroScene as VisualScene}
        submitAction={vi.fn()}
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

    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(submitInput).toHaveBeenCalledTimes(1);
    expect(submitInput).toHaveBeenNthCalledWith(1, {
      playerId: "player_0",
      actionPayload: expect.objectContaining({
        id: "right",
        move: "right",
        hold_ticks: 6,
        metadata: expect.objectContaining({
          input_seq: 1,
          realtime_input: true
        })
      })
    });

    fireEvent.keyDown(window, { key: " " });

    expect(submitInput).toHaveBeenCalledTimes(2);
    expect(submitInput).toHaveBeenNthCalledWith(2, {
      playerId: "player_0",
      actionPayload: expect.objectContaining({
        id: "right_jump",
        move: "right_jump",
        hold_ticks: 6,
        metadata: expect.objectContaining({
          input_seq: 2,
          realtime_input: true
        })
      })
    });
  });
});
