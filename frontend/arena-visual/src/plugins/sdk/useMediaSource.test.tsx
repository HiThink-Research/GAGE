import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

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
});
