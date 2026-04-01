import { describe, expect, it, vi } from "vitest";

import type { ArenaSessionStore } from "../store/arenaSessionStore";
import { usePlaybackControls } from "./usePlaybackControls";

describe("usePlaybackControls", () => {
  it("selectEvent submits a seek command without mutating the selected seq optimistically", async () => {
    const store = {
      setPlaybackMode: vi.fn(),
      setCurrentSceneSeq: vi.fn(),
      loadScene: vi.fn(),
      loadMoreTimeline: vi.fn(),
      submitControl: vi.fn().mockResolvedValue(undefined),
    } as unknown as ArenaSessionStore;

    const controls = usePlaybackControls(store);
    await controls.selectEvent(7);

    expect(store.submitControl).toHaveBeenCalledWith({
      commandType: "seek_seq",
      targetSeq: 7,
    });
    expect(store.setCurrentSceneSeq).not.toHaveBeenCalled();
    expect(store.loadScene).not.toHaveBeenCalled();
  });

  it("submits step, follow-tail, and back-to-tail playback commands through the store", async () => {
    const store = {
      submitControl: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn(),
    } as unknown as ArenaSessionStore;

    const controls = usePlaybackControls(store);

    await controls.stepBackward();
    await controls.stepForward();
    await controls.followTail();
    await controls.backToTail();

    expect(store.submitControl).toHaveBeenNthCalledWith(1, {
      commandType: "step",
      stepDelta: -1,
    });
    expect(store.submitControl).toHaveBeenNthCalledWith(2, {
      commandType: "step",
      stepDelta: 1,
    });
    expect(store.submitControl).toHaveBeenNthCalledWith(3, {
      commandType: "follow_tail",
    });
    expect(store.submitControl).toHaveBeenNthCalledWith(4, {
      commandType: "back_to_tail",
    });
  });

  it("submits the explicit finish command through the store", async () => {
    const store = {
      submitControl: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn(),
    } as unknown as ArenaSessionStore;

    const controls = usePlaybackControls(store);
    await controls.finish();

    expect(store.submitControl).toHaveBeenCalledWith({
      commandType: "finish",
    });
  });

  it("submits the explicit restart command through the store", async () => {
    const store = {
      submitControl: vi.fn().mockResolvedValue(undefined),
      loadMoreTimeline: vi.fn(),
    } as unknown as ArenaSessionStore;

    const controls = usePlaybackControls(store);
    await controls.restart();

    expect(store.submitControl).toHaveBeenCalledWith({
      commandType: "restart",
    });
  });
});
