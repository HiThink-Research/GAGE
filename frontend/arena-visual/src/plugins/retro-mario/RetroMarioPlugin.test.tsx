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
    const submitAction = vi.fn().mockResolvedValue(undefined);

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
        submitAction={submitAction}
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
    expect(submitAction).toHaveBeenCalledWith({
      playerId: "player_0",
      action: {
        id: "right_jump",
        move: "right_jump",
        hold_ticks: 6
      }
    });
  });
});
