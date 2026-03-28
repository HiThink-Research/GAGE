import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ResolvedMediaSource } from "../../gateway/media";
import type { VisualScene } from "../../gateway/types";
import vizdoomScene from "../../test/fixtures/vizdoom.visual.json";
import { VizDoomPlugin } from "./VizDoomPlugin";

const FRAME_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAGUlEQVR4nGNkaGBgYGBg+M8ABYwMjAyMDAwAAB0vAQx0J7s8AAAAAElFTkSuQmCC";

describe("VizDoomPlugin", () => {
  it("renders a non-blank frame, overlay badges, and disables frame actions when input is closed", async () => {
    const submitInput = vi.fn().mockResolvedValue(undefined);

    render(
      <VizDoomPlugin
        session={{
          sessionId: "vizdoom-sample",
          gameId: "vizdoom",
          pluginId: "arena.visualization.vizdoom.frame_v1",
          lifecycle: "closed",
          playback: {
            mode: "paused",
            cursorTs: 3017,
            cursorEventSeq: 17,
            speed: 1,
            canSeek: true
          },
          observer: {
            observerId: "p0",
            observerKind: "player"
          },
          scheduling: {
            family: "real_time_tick",
            phase: "completed",
            acceptsHumanIntent: false,
            activeActorId: "p0"
          },
          capabilities: {},
          summary: {},
          timeline: {}
        }}
        scene={vizdoomScene as VisualScene}
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
    expect(screen.getByRole("heading", { name: /vizdoom frame/i })).toBeInTheDocument();
    expect(screen.getByText("Reward")).toBeInTheDocument();
    expect(screen.getByText("0.75")).toBeInTheDocument();
    expect(screen.getByText("Tick 17. Legal actions: 0, 1, 2")).toBeInTheDocument();

    const fireButton = screen.getByRole("button", { name: /fire/i });
    expect(fireButton).toBeDisabled();
    fireEvent.click(fireButton);
    expect(submitInput).not.toHaveBeenCalled();
  });
});
