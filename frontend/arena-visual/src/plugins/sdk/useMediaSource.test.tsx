import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ResolvedMediaSource } from "../../gateway/media";
import { useMediaSource } from "./useMediaSource";

describe("useMediaSource", () => {
  it("cleans up media subscription on rerender and unmount", () => {
    const unsubscribeA = vi.fn();
    const unsubscribeB = vi.fn();
    const subscribe = vi
      .fn()
      .mockReturnValueOnce(unsubscribeA)
      .mockReturnValueOnce(unsubscribeB);

    const { rerender, unmount } = renderHook(
      ({ mediaId }) =>
        useMediaSource({
          sessionId: "sample-1",
          mediaId,
          subscribe,
        }),
      { initialProps: { mediaId: "frame-1" } },
    );

    act(() => {
      rerender({ mediaId: "frame-2" });
    });

    expect(unsubscribeA).toHaveBeenCalledTimes(1);
    expect(subscribe).toHaveBeenCalledTimes(2);

    unmount();
    expect(unsubscribeB).toHaveBeenCalledTimes(1);
  });

  it("keeps the previous ready frame while the next media id is still loading", () => {
    const listeners = new Map<string, (state: ResolvedMediaSource) => void>();
    const subscribe = vi.fn((request, listener: (state: ResolvedMediaSource) => void) => {
      listeners.set(request.mediaId, listener);
      if (request.mediaId === "frame-1") {
        listener({
          mediaId: request.mediaId,
          status: "ready",
          src: "data:image/png;base64,frame-1",
        });
      } else {
        listener({
          mediaId: request.mediaId,
          status: "loading",
        });
      }
      return () => {
        listeners.delete(request.mediaId);
      };
    });

    const { result, rerender } = renderHook(
      ({ mediaId }) =>
        useMediaSource({
          sessionId: "sample-1",
          mediaId,
          subscribe,
        }),
      { initialProps: { mediaId: "frame-1" } },
    );

    expect(result.current).toEqual({
      mediaId: "frame-1",
      status: "ready",
      src: "data:image/png;base64,frame-1",
    });

    act(() => {
      rerender({ mediaId: "frame-2" });
    });

    expect(result.current).toEqual({
      mediaId: "frame-1",
      status: "ready",
      src: "data:image/png;base64,frame-1",
    });

    act(() => {
      listeners.get("frame-2")?.({
        mediaId: "frame-2",
        status: "ready",
        src: "data:image/png;base64,frame-2",
      });
    });

    expect(result.current).toEqual({
      mediaId: "frame-2",
      status: "ready",
      src: "data:image/png;base64,frame-2",
    });
  });
});
