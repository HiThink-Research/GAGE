import { describe, expect, it, vi } from "vitest";

import type { ArenaSessionStore } from "../store/arenaSessionStore";
import { usePlaybackControls } from "./usePlaybackControls";

describe("usePlaybackControls", () => {
  it("selectEvent only updates the selected seq and leaves scene loading to the page effect", async () => {
    const store = {
      setPlaybackMode: vi.fn(),
      setCurrentSceneSeq: vi.fn(),
      loadScene: vi.fn(),
      loadMoreTimeline: vi.fn(),
    } as unknown as ArenaSessionStore;

    const controls = usePlaybackControls(store);
    await controls.selectEvent(7);

    expect(store.setCurrentSceneSeq).toHaveBeenCalledWith(7);
    expect(store.loadScene).not.toHaveBeenCalled();
  });
});
