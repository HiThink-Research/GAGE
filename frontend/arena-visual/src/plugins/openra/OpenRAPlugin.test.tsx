import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ResolvedMediaSource } from "../../gateway/media";
import type { VisualScene } from "../../gateway/types";
import openraScene from "../../test/fixtures/openra.visual.json";
import { OpenRAPlugin } from "./OpenRAPlugin";

const FRAME_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAGUlEQVR4nGNkaGBgYGBg+M8ABYwMjAyMDAwAAB0vAQx0J7s8AAAAAElFTkSuQmCC";

function markSceneAsNativeRuntime(scene: VisualScene): void {
  const body = scene.body;
  if (typeof body !== "object" || body === null) {
    return;
  }
  const rts = (body as Record<string, unknown>).rts;
  if (typeof rts !== "object" || rts === null) {
    return;
  }
  const map = (rts as Record<string, unknown>).map;
  if (typeof map !== "object" || map === null) {
    return;
  }
  (map as Record<string, unknown>).previewSource = "native_runtime";
}

function buildNativeScene(): VisualScene {
  const nativeScene = JSON.parse(JSON.stringify(openraScene)) as VisualScene;
  if (!Array.isArray(nativeScene.legalActions)) {
    nativeScene.legalActions = [];
  }
  nativeScene.legalActions.push({
    id: "bridge_input",
    label: "Native input",
    text: "Native input",
    payloadSchema: {
      event_type: "mouse_down",
      x: 0,
      y: 0,
    },
  });
  markSceneAsNativeRuntime(nativeScene);
  return nativeScene;
}

function installLowLatencyCanvasMocks() {
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

  return {
    drawImage,
    fetchMock,
    restore() {
      getContextMock.mockRestore();
      vi.unstubAllGlobals();
    },
  };
}

describe("OpenRAPlugin", () => {
  it("renders the RTS stage, frame image, strategic panels, and legal action chips", async () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);

    render(
      <OpenRAPlugin
        session={{
          sessionId: "openra-sample",
          gameId: "openra",
          pluginId: "arena.visualization.openra.rts_v1",
          lifecycle: "live_running",
          playback: {
            mode: "paused",
            cursorTs: 5031,
            cursorEventSeq: 31,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "player_0",
            observerKind: "player",
          },
          scheduling: {
            family: "real_time_tick",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={openraScene as VisualScene}
        submitAction={vi.fn()}
        submitInput={submitInput}
        mediaSubscribe={(request, listener) => {
          listener({
            mediaId: request.mediaId,
            status: "ready",
            src: FRAME_DATA_URL,
          } as ResolvedMediaSource);
          return () => {};
        }}
        isFallback={false}
      />,
    );

    await waitFor(() =>
      expect(screen.getByTestId("openra-map-image")).toHaveAttribute("src", FRAME_DATA_URL),
    );

    expect(screen.getByTestId("openra-map-markers")).toBeInTheDocument();
    expect(screen.getByTestId("openra-map-marker-mcv_1")).toBeInTheDocument();
    expect(screen.getByTestId("openra-map-marker-rifle_2")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /openra rts/i })).toBeInTheDocument();
    expect(screen.getByTestId("openra-stage")).toHaveTextContent("Credits 1200");
    expect(screen.getByTestId("openra-stage")).toHaveTextContent("Power +10");
    expect(screen.getByTestId("openra-stage")).toHaveTextContent("Hold the ridge");
    expect(screen.getByTestId("openra-stage")).toHaveTextContent("Map preview · Marigold Town");
    expect(screen.getByTestId("openra-stage")).toHaveTextContent("Rifle Infantry");
    expect(screen.getByTestId("openra-stage")).toHaveTextContent("Barracks");
    expect(screen.getByTestId("openra-actions")).toHaveTextContent("Select units");

    fireEvent.click(screen.getByRole("button", { name: /issue command/i }));
    expect(submitInput).toHaveBeenCalledWith({
      playerId: "player_0",
      actionPayload: {
        move: "issue_command",
        payload: {
          command: "attack_move",
          target: {
            x: 18,
            y: 11,
          },
        },
      },
    });
  });

  it("submits native bridge pointer events from the rendered frame surface", async () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);
    const nativeScene = buildNativeScene();

    render(
      <OpenRAPlugin
        session={{
          sessionId: "openra-sample",
          gameId: "openra",
          pluginId: "arena.visualization.openra.rts_v1",
          lifecycle: "live_running",
          playback: {
            mode: "paused",
            cursorTs: 5031,
            cursorEventSeq: 31,
            speed: 1,
            canSeek: true,
          },
          observer: {
            observerId: "player_0",
            observerKind: "player",
          },
          scheduling: {
            family: "real_time_tick",
            phase: "waiting_for_intent",
            acceptsHumanIntent: true,
            activeActorId: "player_0",
          },
          capabilities: {},
          summary: {},
          timeline: {},
        }}
        scene={nativeScene}
        submitAction={vi.fn()}
        submitInput={submitInput}
        mediaSubscribe={(request, listener) => {
          listener({
            mediaId: request.mediaId,
            status: "ready",
            src: FRAME_DATA_URL,
          } as ResolvedMediaSource);
          return () => {};
        }}
        isFallback={false}
      />,
    );

    const surface = await screen.findByTestId("openra-native-surface");
    const nativeImage = await screen.findByTestId("openra-map-image");
    expect(nativeImage).toHaveClass("openra-stage__image--native");
    Object.defineProperty(surface, "getBoundingClientRect", {
      value: () => ({
        left: 20,
        top: 10,
        width: 640,
        height: 360,
        right: 660,
        bottom: 370,
        x: 20,
        y: 10,
        toJSON: () => ({}),
      }),
    });

    fireEvent.mouseDown(surface, {
      button: 0,
      buttons: 1,
      clientX: 180,
      clientY: 100,
    });

    expect(submitInput).toHaveBeenCalledWith({
      playerId: "player_0",
      actionPayload: {
        move: "bridge_input",
        payload: {
          event_type: "mouse_down",
          button: "left",
          buttons: ["left"],
          x: 320,
          y: 180,
          viewport: {
            width: 1280,
            height: 720,
          },
          modifiers: [],
        },
        metadata: {
          input_seq: 1,
          realtime_input: true,
        },
      },
    });
  });

  it("renders low-latency native streams through the shared multipart canvas path", async () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);
    const nativeScene = buildNativeScene();
    const { drawImage, fetchMock, restore } = installLowLatencyCanvasMocks();

    try {
      render(
        <OpenRAPlugin
          session={{
            sessionId: "openra-sample",
            gameId: "openra",
            pluginId: "arena.visualization.openra.rts_v1",
            lifecycle: "live_running",
            playback: {
              mode: "paused",
              cursorTs: 5031,
              cursorEventSeq: 31,
              speed: 1,
              canSeek: true,
            },
            observer: {
              observerId: "player_0",
              observerKind: "player",
            },
            scheduling: {
              family: "real_time_tick",
              phase: "waiting_for_intent",
              acceptsHumanIntent: true,
              activeActorId: "player_0",
            },
            capabilities: {},
            summary: {},
            timeline: {},
          }}
          scene={nativeScene}
          submitAction={vi.fn()}
          submitInput={submitInput}
          mediaSubscribe={(request, listener) => {
            listener({
              mediaId: request.mediaId,
              status: "ready",
              src: "http://arena.local/openra/stream",
              ref: {
                mediaId: request.mediaId,
                transport: "low_latency_channel",
                mimeType: "multipart/x-mixed-replace",
                url: "http://arena.local/openra/stream",
              },
            } as ResolvedMediaSource);
            return () => {};
          }}
          isFallback={false}
        />,
      );

      await waitFor(() =>
        expect(screen.getByTestId("openra-map-canvas")).toBeInTheDocument(),
      );
      await waitFor(() => expect(fetchMock).toHaveBeenCalled());
      expect(String(fetchMock.mock.calls[0]?.[0] ?? "")).toContain("http://arena.local/openra/stream");
      expect(drawImage).toHaveBeenCalled();
      expect(screen.queryByTestId("openra-map-image")).not.toBeInTheDocument();
    } finally {
      restore();
    }
  });

  it("transitions from waiting state to native stream view", async () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);
    const nativeScene = buildNativeScene();
    const { restore } = installLowLatencyCanvasMocks();

    try {
      const view = render(
        <OpenRAPlugin
          session={{
            sessionId: "openra-sample",
            gameId: "openra",
            pluginId: "arena.visualization.openra.rts_v1",
            lifecycle: "live_running",
            playback: {
              mode: "paused",
              cursorTs: 5031,
              cursorEventSeq: 31,
              speed: 1,
              canSeek: true,
            },
            observer: {
              observerId: "player_0",
              observerKind: "player",
            },
            scheduling: {
              family: "real_time_tick",
              phase: "waiting_for_intent",
              acceptsHumanIntent: true,
              activeActorId: "player_0",
            },
            capabilities: {},
            summary: {},
            timeline: {},
          }}
          scene={undefined}
          submitAction={vi.fn()}
          submitInput={submitInput}
          mediaSubscribe={() => () => {}}
          isFallback={false}
        />,
      );

      expect(screen.getByText(/rts scene unavailable/i)).toBeInTheDocument();

      view.rerender(
        <OpenRAPlugin
          session={{
            sessionId: "openra-sample",
            gameId: "openra",
            pluginId: "arena.visualization.openra.rts_v1",
            lifecycle: "live_running",
            playback: {
              mode: "paused",
              cursorTs: 5031,
              cursorEventSeq: 31,
              speed: 1,
              canSeek: true,
            },
            observer: {
              observerId: "player_0",
              observerKind: "player",
            },
            scheduling: {
              family: "real_time_tick",
              phase: "waiting_for_intent",
              acceptsHumanIntent: true,
              activeActorId: "player_0",
            },
            capabilities: {},
            summary: {},
            timeline: {},
          }}
          scene={nativeScene}
          submitAction={vi.fn()}
          submitInput={submitInput}
          mediaSubscribe={(request, listener) => {
            listener({
              mediaId: request.mediaId,
              status: "ready",
              src: "http://arena.local/openra/stream",
              ref: {
                mediaId: request.mediaId,
                transport: "low_latency_channel",
                mimeType: "multipart/x-mixed-replace",
                url: "http://arena.local/openra/stream",
              },
            } as ResolvedMediaSource);
            return () => {};
          }}
          isFallback={false}
        />,
      );

      await waitFor(() =>
        expect(screen.getByTestId("openra-map-canvas")).toBeInTheDocument(),
      );
    } finally {
      restore();
    }
  });
});
