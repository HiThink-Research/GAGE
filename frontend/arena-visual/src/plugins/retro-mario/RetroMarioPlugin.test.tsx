import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ResolvedMediaSource } from "../../gateway/media";
import type { VisualScene } from "../../gateway/types";
import retroScene from "../../test/fixtures/retro-mario.visual.json";
import { RetroMarioPlugin } from "./RetroMarioPlugin";

const FRAME_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAGUlEQVR4nGNkaGBgYGBg+M8ABYwMjAyMDAwAAB0vAQx0J7s8AAAAAElFTkSuQmCC";

describe("RetroMarioPlugin", () => {
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

  it("submits stable retro action ids with hold duration metadata", async () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);

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

    fireEvent.click(screen.getByRole("button", { name: /right \+ jump/i }));
    expect(submitInput).toHaveBeenCalledWith({
      playerId: "player_0",
      actionPayload: {
        id: "right_jump",
        move: "right_jump",
        hold_ticks: 6
      }
    });
  });

  it("maps keyboard state into retro actions and resubmits while keys stay held across frames", async () => {
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
      actionPayload: {
        id: "right",
        move: "right",
        hold_ticks: 6
      }
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

    await waitFor(() => expect(submitInput).toHaveBeenCalledTimes(2));
    expect(submitInput).toHaveBeenLastCalledWith({
      playerId: "player_0",
      actionPayload: {
        id: "right",
        move: "right",
        hold_ticks: 6
      }
    });

    fireEvent.keyDown(window, { key: " " });
    expect(submitInput).toHaveBeenLastCalledWith({
      playerId: "player_0",
      actionPayload: {
        id: "right_jump",
        move: "right_jump",
        hold_ticks: 6
      }
    });

    fireEvent.keyUp(window, { key: " " });
    expect(submitInput).toHaveBeenLastCalledWith({
      playerId: "player_0",
      actionPayload: {
        id: "right",
        move: "right",
        hold_ticks: 6
      }
    });

    fireEvent.keyUp(window, { key: "ArrowRight" });
    expect(submitInput).toHaveBeenLastCalledWith({
      playerId: "player_0",
      actionPayload: {
        id: "noop",
        move: "noop"
      }
    });
  });
});
